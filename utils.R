################################################################################
# Summaries and loaders

# Convenience function for averaging the value column within groups, with a
# 95% confidence interval from the sample standard deviation
summarize_ci <- function(df, ...) {
  df %>%
    group_by(...) %>%
    summarize(
      n = n(),
      std.err = sd(value) / sqrt(n),
      mean = mean(value),
      .groups = "drop"
    ) %>%
    mutate(
      lo = mean - 1.96 * std.err,
      hi = mean + 1.96 * std.err
    )
}

# Convenience function for writing a trained weight as a short label, with the
# lexicographic arms as an infinity sign
format_weight <- function(weight, signed = FALSE) {
  case_when(
    is.na(weight)          ~ "NA",
    weight == Inf & signed ~ "+∞",
    weight == Inf          ~ "∞",
    weight == -Inf         ~ "−∞",
    .default = as.character(weight)
  )
}

# Convenience function for averaging the capability scores of the policies
# trained under a reward, along with the dose policies and the untrained base,
# within each policy with a Wald binomial 95% interval
summarize_capability <- function(policy_capability, reward, metrics) {
  # The untrained base is one run per model, whatever the reward
  # NOTE: gpt-oss's policies answer with the analysis channel disabled, so its
  #       base is the reasoning-off run rather than the default base.
  base_reward <- c(
    "gemma"  = "free",
    "qwen"   = "free",
    "gptoss" = "base_reasoning_off"
  )

  policy_capability %>%
    filter(
      metric %in% metrics,
      (arm_type %in% c("grid", "syc") & reward == .env$reward) |
        (arm_type == "dose" & reward == "discrimination") |
        (arm_type == "base" & reward == base_reward[model])
    ) %>%
    group_by(
      model,
      metric,
      arm,
      arm_type,
      beta_help,
      beta_lean,
      beta_syc,
      rm,
      q
    ) %>%
    summarize(
      n = n(),
      value = mean(value),
      .groups = "drop"
    ) %>%
    mutate(
      se = sqrt(value * (1 - value) / n),
      lo = pmax(0, value - 1.96 * se),
      hi = pmin(1, value + 1.96 * se)
    )
}

################################################################################
# Rubric layout

# Heights of the family and subsection rows and the gap between them
FAMILY_HEIGHT <- 0.72
SUBSECTION_HEIGHT <- 1.68
ROW_GAP <- 0.05

# Heights, as shares of the family row, of the family names and of the counts
# below them
FAMILY_LABEL_Y <- 0.62
FAMILY_COUNT_Y <- 0.28

# Distance below the subsection row of the legend of small families, and the
# half-height of its color swatches
LEGEND_OFFSET <- 0.22
LEGEND_SWATCH_HALF_HEIGHT <- 0.06

# Families with fewer items than this are listed in the legend rather than
# labeled in place
MIN_FAMILY_ITEMS_LABELED <- 20

# Text sizes (in mm) of the family names, of the subsection labels, and of the
# legend and count labels
FAMILY_LABEL_SIZE <- 2.6
SUBSECTION_LABEL_SIZE <- 2
LEGEND_TEXT_SIZE <- 2.1

# Gap, in rubric items, on each side of the slash between a family's kept and
# total counts
COUNT_GAP <- 0.8

# Opacity of the subsection row and of the shaded share of it the fit kept
SUBSECTION_ALPHA <- 0.22
KEPT_ALPHA <- 0.52

# Widths, in rubric items, of one legend entry, of its color swatch, of the
# gap between the swatch and the family name, and of the gaps before the
# number of items kept and before the total
LEGEND_ENTRY_WIDTH <- 52
LEGEND_SWATCH_WIDTH <- 4
LEGEND_TEXT_GAP <- 6
LEGEND_KEPT_GAP <- 3
LEGEND_TOTAL_GAP <- 1.2

# The longest subsection label drawn in place
MAX_LABEL_CHARACTERS <- 24

# Convenience function for plotting the rubric as a family row over a
# subsection row, with the number of items kept by the fit over the number
# of items in each family and, if shaded, the share of each subsection the
# fit kept
plot_rubric <- function(rubric_structure, shaded, width, file) {
  # The y position of the legend that lists the small families below the rows
  legend_y <- -ROW_GAP - SUBSECTION_HEIGHT - LEGEND_OFFSET

  # The x axis spans the rubric items over the figure width. A rotated
  # subsection label needs its text size (in points) plus 2pt of padding, and
  # legend text is about 3pt per character.
  # NOTE: The width is converted at 72 PostScript points per inch rather than
  #       72.27 TeX points; the difference is immaterial at this scale.
  items_per_point <- sum(rubric_structure$n_items) / (width * 72)
  min_label_width <- (SUBSECTION_LABEL_SIZE * .pt + 2) * items_per_point
  character_width <- 3 * items_per_point

  # Lay the families out along the x axis, largest first
  families <- rubric_structure %>%
    group_by(family) %>%
    summarize(
      n_items = sum(n_items),
      n_kept = sum(n_kept),
      .groups = "drop"
    ) %>%
    arrange(desc(n_items)) %>%
    mutate(
      xmax = cumsum(n_items),
      xmin = xmax - n_items,
      label = str_to_sentence(family),
      n_items_label = glue("/ {n_items}")
    )

  # Lay the subsections out under their families, largest first
  subsections <- rubric_structure %>%
    mutate(family = factor(family, levels = families$family)) %>%
    arrange(family, desc(n_items)) %>%
    mutate(
      xmax = cumsum(n_items),
      xmin = xmax - n_items,
      share_kept = n_kept / n_items,
      label = str_to_sentence(str_replace_all(subsection, "_", " ")),
      label_fits = n_items >= min_label_width &
        nchar(label) <= MAX_LABEL_CHARACTERS
    )

  # The families too narrow to label in place are listed in a legend below
  families_legend <- families %>%
    filter(n_items < MIN_FAMILY_ITEMS_LABELED) %>%
    mutate(
      legend_x = (row_number() - 1) * LEGEND_ENTRY_WIDTH,
      kept_x = legend_x + LEGEND_TEXT_GAP + nchar(label) * character_width +
        LEGEND_KEPT_GAP,
      total_x = kept_x + nchar(as.character(n_kept)) * character_width +
        LEGEND_TOTAL_GAP
    )

  # The families wide enough to be labeled in place
  families_labeled <- families %>%
    filter(n_items >= MIN_FAMILY_ITEMS_LABELED)

  # The share of each subsection that the fit kept, drawn as a darker band
  # rising from the bottom of the subsection row
  kept_layer <- if (shaded) {
    geom_rect(
      aes(
        xmin = xmin,
        xmax = xmax,
        ymin = -ROW_GAP - SUBSECTION_HEIGHT,
        ymax = -ROW_GAP - SUBSECTION_HEIGHT * (1 - share_kept),
        fill = family
      ),
      data = subsections,
      color = NA,
      alpha = KEPT_ALPHA
    )
  }

  p_rubric <- ggplot() +
    # NOTE: The row heights are mapped rather than set so that the y axis is
    #       trained on them.
    geom_rect(
      aes(
        xmin = xmin,
        xmax = xmax,
        ymin = 0,
        ymax = FAMILY_HEIGHT,
        fill = family
      ),
      data = families,
      color = "white"
    ) +
    geom_rect(
      aes(
        xmin = xmin,
        xmax = xmax,
        ymin = -ROW_GAP - SUBSECTION_HEIGHT,
        ymax = -ROW_GAP,
        fill = family
      ),
      data = subsections,
      color = "white",
      alpha = SUBSECTION_ALPHA
    ) +
    kept_layer +
    geom_text(
      aes(
        x = (xmin + xmax) / 2,
        y = FAMILY_HEIGHT * FAMILY_LABEL_Y,
        label = label
      ),
      data = families_labeled,
      color = "white",
      size = FAMILY_LABEL_SIZE
    ) +
    geom_text(
      aes(
        x = (xmin + xmax) / 2 - COUNT_GAP,
        y = FAMILY_HEIGHT * FAMILY_COUNT_Y,
        label = n_kept
      ),
      data = families_labeled,
      color = "white",
      size = LEGEND_TEXT_SIZE,
      fontface = "bold",
      hjust = 1
    ) +
    geom_text(
      aes(
        x = (xmin + xmax) / 2 + COUNT_GAP,
        y = FAMILY_HEIGHT * FAMILY_COUNT_Y,
        label = n_items_label
      ),
      data = families_labeled,
      color = "white",
      size = LEGEND_TEXT_SIZE,
      hjust = 0
    ) +
    geom_rect(
      aes(
        xmin = legend_x,
        xmax = legend_x + LEGEND_SWATCH_WIDTH,
        ymin = legend_y - LEGEND_SWATCH_HALF_HEIGHT,
        ymax = legend_y + LEGEND_SWATCH_HALF_HEIGHT,
        fill = family
      ),
      data = families_legend
    ) +
    geom_text(
      aes(
        x = legend_x + LEGEND_TEXT_GAP,
        y = legend_y,
        label = label
      ),
      data = families_legend,
      hjust = 0,
      size = LEGEND_TEXT_SIZE
    ) +
    geom_text(
      aes(
        x = kept_x,
        y = legend_y,
        label = n_kept
      ),
      data = families_legend,
      hjust = 0,
      size = LEGEND_TEXT_SIZE,
      fontface = "bold"
    ) +
    geom_text(
      aes(
        x = total_x,
        y = legend_y,
        label = n_items_label
      ),
      data = families_legend,
      hjust = 0,
      size = LEGEND_TEXT_SIZE
    ) +
    geom_text(
      aes(
        x = (xmin + xmax) / 2,
        y = -ROW_GAP - SUBSECTION_HEIGHT / 2,
        label = label
      ),
      data = filter(subsections, label_fits),
      angle = 90,
      size = SUBSECTION_LABEL_SIZE,
      color = "grey30"
    ) +
    scale_fill_brewer(palette = "Dark2", guide = "none") +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = expansion(add = c(0.06, 0))) +
    theme_void() +
    theme(plot.margin = margin(2, 2, 2, 2))

  ggsave(
    path("figures", file),
    plot = p_rubric,
    width = width,
    height = 1.75,
    device = cairo_pdf
  )
}
