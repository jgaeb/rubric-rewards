source("setup.R")
library(patchwork)

################################################################################
# Plot dimensions

# The text width of the paper, in inches
TEXT_WIDTH <- 397.485 / 72.27

################################################################################
# Translators for shorthand

TR_MODEL <- c(
  "gemma"  = "Gemma 3 12B",
  "qwen"   = "Qwen3 14B",
  "gptoss" = "GPT-OSS 20B"
)

TR_SPLIT <- c(
  "eval_in"  = "In Distribution",
  "eval_out" = "Out of Distribution"
)

TR_RM <- c(
  "rubric"             = "Rubric (free)",
  "rubric_constrained" = "Rubric (constr.)",
  "skywork8b"          = "Skywork 8B",
  "skywork3b"          = "Skywork 3B",
  "skywork1b"          = "Skywork 1B",
  "qrm"                = "QRM 27B",
  "deberta"            = "DeBERTa"
)

TR_METRIC <- c(
  "mmlu_acc"      = "MMLU",
  "ifeval_strict" = "IFEval (strict)"
)

# Frontier models labeled on the political slant plot
TR_FLEET_SLANT <- c(
  "grok45"      = "Grok 4.5",
  "luna"        = "GPT-5.6 Luna",
  "sonnet5"     = "Sonnet 5",
  "gemini31pro" = "Gemini 3.1 Pro",
  "llama70b"    = "Llama 70B"
)

# Frontier models labeled on the helpfulness / harmlessness frontier
TR_FLEET_FRONTIER <- c(
  "gptoss20b"    = "GPT-OSS 20B",
  "gemini3flash" = "Gemini 3 Flash",
  "gemma27b"     = "Gemma 27B",
  "luna"         = "GPT-5.6 Luna",
  "sonnet5"      = "Sonnet 5",
  "gemini31pro"  = "Gemini 3.1 Pro"
)

# Frontier models labeled on the sycophancy plot
TR_FLEET_SYCOPHANCY <- c(
  "gemma12b"   = "Gemma 12B",
  "gemma27b"   = "Gemma 27B",
  "llama70b"   = "Llama 70B",
  "mistral24b" = "Mistral 24B",
  "luna"       = "GPT-5.6 Luna",
  "gptoss20b"  = "GPT-OSS 20B",
  "maverick"   = "Llama 4 Maverick",
  "grok45"     = "Grok 4.5",
  "qwen7b"     = "Qwen 7B",
  "sonnet5"    = "Sonnet 5"
)

# Reward models on the agreement plot, with each neural model's position on
# the RewardBench 2 leaderboard (71 models ranked), which is also the order the
# axes run in. Both rubric fits are labeled "Rubric"; only the one the
# policies were trained on is kept.
TR_RM_RANKED <- c(
  "rubric"             = "Rubric",
  "rubric_constrained" = "Rubric",
  "skywork8b"          = "Skywork 8B (#1)",
  "qrm"                = "QRM 27B (#8)",
  "skywork3b"          = "Skywork 3B (#16)",
  "skywork1b"          = "Skywork 1B (#40)",
  "deberta"            = "DeBERTa (#65)"
)

# Evaluations on the agreement plot
TR_EVAL <- c(
  "feedback"  = "Sycophancy",
  "harm"      = "Harm / helpfulness",
  "political" = "Political",
  "jobs"      = "Discrimination"
)

# Convenience function for putting discrete covariates in a consistent order
relabel <- function(translator) {
  function(s) {
    factor(translator[s], levels = translator)
  }
}

# Colors for the base models
MODEL_COLORS <- c(
  "Gemma 3 12B" = "#D55E00",
  "Qwen3 14B"   = "#0072B2",
  "GPT-OSS 20B" = "#009E73"
)

# Colors for the lean weights (blue to red) on the non-interference plot
LEAN_COLORS <- c(
  "#053061",
  "#2166ac",
  "#67a9cf",
  "#4d4d4d",
  "#ef8a62",
  "#b2182b",
  "#67001f"
)

# Colors for the help weights (light teal to navy) on the non-interference plot
HELP_COLORS <- c(
  "#7fcdbb",
  "#41b6c4",
  "#1d91c0",
  "#225ea8",
  "#0c2c84"
)

################################################################################
# Load the data

# Load the rubric, one row per subsection
rubric_structure <- path("tables", "rubric_structure.parquet") %>%
  read_parquet() %>%
  as_tibble()

# Load the judge scores of the trained policies, one row per (arm, prompt,
# judge metric)
policy_judge <- path("tables", "policy_judge_obs.parquet") %>%
  read_parquet() %>%
  as_tibble()

# Load the judge scores of the frontier models on the same prompts
fleet_judge <- path("tables", "fleet_judge_obs.parquet") %>%
  read_parquet() %>%
  as_tibble() %>%
  rename(model = fleet_model)

# Load the hiring picks of the dose policies, one row per prompt
hiring_picks <- path("tables", "disc_jobs_obs.parquet") %>%
  read_parquet() %>%
  as_tibble()

# Load the movie picks of the dose policies, one row per name pair
movie_picks <- path("tables", "disc_movies_obs.parquet") %>%
  read_parquet() %>%
  as_tibble()

# Load the sycophancy scores of the trained policies
policy_sycophancy <- path("tables", "policy_syc_obs.parquet") %>%
  read_parquet() %>%
  as_tibble()

# Load the sycophancy scores of the frontier models on the same prompts
fleet_sycophancy_obs <- path("tables", "fleet_syc_obs.parquet") %>%
  read_parquet() %>%
  as_tibble() %>%
  rename(model = fleet_model)

# Load the correlations between reward models, one row per unordered pair
rm_correlations <- path("tables", "rm_correlations.parquet") %>%
  read_parquet() %>%
  as_tibble()

# Load the capability scores of the policies, one row per (arm, question)
policy_capability <- path("tables", "policy_capability_obs.parquet") %>%
  read_parquet() %>%
  as_tibble()

# Load the capability scores of the policies trained under the reward,
# averaged within each policy, with the steering-grid and sycophancy policies
# labeled by their weights
capability <- policy_capability %>%
  summarize_capability(reward_regime, metrics = names(TR_METRIC)) %>%
  mutate(
    model = relabel(TR_MODEL)(model),
    metric = relabel(TR_METRIC)(metric),
    family = if_else(arm_type == "syc", "Sycophancy", "Steering\ngrid"),
    family = factor(family, levels = c("Steering\ngrid", "Sycophancy")),
    label = if_else(
      arm_type == "syc",
      format_weight(beta_syc),
      str_c(
        format_weight(beta_help),
        " / ",
        format_weight(beta_lean, signed = TRUE)
      )
    )
  )

################################################################################
################################ MAIN TEXT PLOTS ###############################
################################################################################

################################################################################
# The rubric

# Plot the rubric as a family row over a subsection row, with the number of
# items kept by the fit over the number of items in each family
plot_rubric(
  rubric_structure,
  shaded = FALSE,
  width = TEXT_WIDTH,
  file = "rubric.pdf"
)

write_csv(rubric_structure, path("figures", "rubric_data.csv"))

################################################################################
# Reward-model agreement

# Rank correlations between every pair of reward models over the same
# responses. The table holds one row per unordered pair, so it is mirrored to
# fill the square, and the diagonal is left empty.
rm_correlations_mirrored <- rm_correlations %>%
  rename(a = b, b = a)

# The rubric reward model the policies were trained on
rubric_rm <- if (reward_regime == "free") "rubric" else "rubric_constrained"

reward_models <- c(
  rubric_rm,
  "skywork8b",
  "qrm",
  "skywork3b",
  "skywork1b",
  "deberta"
)

rm_correlation <- rm_correlations %>%
  bind_rows(rm_correlations_mirrored) %>%
  filter(
    a != b,
    a %in% reward_models,
    b %in% reward_models
  ) %>%
  distinct(model, lever, a, b, .keep_all = TRUE) %>%
  mutate(
    model = relabel(TR_MODEL)(model),
    a = factor(TR_RM_RANKED[a], levels = unique(TR_RM_RANKED)),
    b = factor(TR_RM_RANKED[b], levels = rev(unique(TR_RM_RANKED))),
    eval = relabel(TR_EVAL)(lever)
  )

# Plot the agreement on the given evaluations, with a line separating the
# rubric row and column from the neural block, one column per model and, for
# more than one evaluation, one row per evaluation
plot_rm_correlation <- function(df, evals, file) {
  multi <- length(evals) > 1
  facet_layer <- if (multi) {
    facet_grid(rows = vars(eval), cols = vars(model))
  } else {
    facet_wrap(vars(model))
  }
  height <- if (multi) 7.2 else 2.9

  p_rm_correlation <- df %>%
    filter(lever %in% evals) %>%
    mutate(
      # Correlations are written without a leading zero, with a true minus
      # sign, and with no sign on a correlation that rounds to zero
      spearman_rounded = round(spearman, 2),
      spearman_rounded = if_else(spearman_rounded == 0, 0, spearman_rounded),
      spearman_label = formatC(spearman_rounded, format = "f", digits = 2),
      spearman_label = str_replace(spearman_label, "^(-?)0\\.", "\\1."),
      spearman_label = str_replace(spearman_label, "^-", "−"),
      dark_cell = spearman > 0.55
    ) %>%
    ggplot(aes(x = a, y = b, fill = spearman)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_vline(xintercept = 1.5, color = "grey20", linewidth = 0.5) +
    geom_hline(
      yintercept = nlevels(df$b) - 0.5,
      color = "grey20",
      linewidth = 0.5
    ) +
    geom_text(aes(label = spearman_label, color = dark_cell), size = 1.9) +
    scale_fill_distiller(
      palette = "YlGnBu",
      direction = 1,
      limits = c(0, 1),
      breaks = c(0, 0.5, 1),
      na.value = "grey92",
      guide = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(7, "lines"),
        barheight = unit(0.4, "lines")
      )
    ) +
    scale_color_manual(
      values = c("TRUE" = "white", "FALSE" = "grey15"),
      guide = "none"
    ) +
    coord_fixed() +
    labs(
      x = NULL,
      y = NULL,
      fill = "Spearman"
    ) +
    facet_layer +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 6),
      axis.text.y = element_text(size = 6),
      panel.grid = element_blank(),
      legend.position = "bottom",
      legend.title = element_text(size = 7),
      legend.text = element_text(size = 7),
      legend.key.size = unit(0.6, "lines"),
      legend.box.spacing = unit(2, "pt")
    )

  ggsave(
    path("figures", file),
    plot = p_rm_correlation,
    width = TEXT_WIDTH,
    height = height,
    device = cairo_pdf
  )
}

plot_rm_correlation(
  rm_correlation,
  evals = "harm",
  file = glue("rm_correlation{file_suffix}.pdf")
)

write_csv(
  rm_correlation,
  path("figures", glue("rm_correlation{file_suffix}_data.csv"))
)

################################################################################
# Steering

########## Political slant

# Average slant of each frontier model, colored from liberal to conservative
# NOTE: The tick color is precomputed because both the color and fill
#       scales of this plot are taken by the model.
fleet_slant <- fleet_judge %>%
  filter(eval == "political", metric == "political_slant") %>%
  summarize_ci(model) %>%
  mutate(
    color = pal_div_gradient("darkblue", "grey85", "darkred")(
      rescale(mean, from = c(-2, 2))
    ),
    label = TR_FLEET_SLANT[model]
  )

# Average slant of the policies trained on each lean weight. The lexicographic
# arms (beta_lean = ±∞) sit past the finite grid at ±5.
steering_slant <- policy_judge %>%
  filter(
    reward == reward_regime,
    eval == "political",
    metric == "political_slant",
    beta_help == 0
  ) %>%
  summarize_ci(model, beta_lean, split) %>%
  mutate(
    model = relabel(TR_MODEL)(model),
    split = relabel(TR_SPLIT)(split),
    x = pmax(pmin(beta_lean, 5), -5)
  )

# Plot the slant against the lean weight, with the frontier models as ticks in
# the right margin
p_slant <- steering_slant %>%
  ggplot(aes(
    x = x,
    y = mean,
    color = model,
    group = interaction(model, split)
  )) +
  geom_ribbon(
    aes(ymin = lo, ymax = hi, fill = model),
    color = NA,
    alpha = 0.2
  ) +
  geom_line(aes(linetype = split)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = c(-4, 4), linetype = "dotted") +
  geom_segment(
    aes(y = mean, yend = mean),
    data = fleet_slant,
    inherit.aes = FALSE,
    x = 5.01,
    xend = 5.42,
    color = fleet_slant$color,
    linewidth = 0.7
  ) +
  geom_text(
    aes(y = mean, label = label),
    data = drop_na(fleet_slant, label),
    inherit.aes = FALSE,
    x = 5.5,
    hjust = 0,
    color = "grey25",
    size = 1.8
  ) +
  scale_x_continuous(
    limits = c(-5, 5),
    expand = c(0, 0),
    oob = oob_keep,
    breaks = c(-5, -4, -3, -1, 0, 1, 3, 4, 5),
    labels = c("−∞", "···", "−3", "−1", "0", "1", "3", "···", "∞")
  ) +
  scale_color_manual(values = MODEL_COLORS, aesthetics = c("color", "fill")) +
  coord_cartesian(clip = "off") +
  labs(
    x = expr(beta["lean"]),
    y = expr(phantom(0) %<-% "Liberal" ~~ "Conservative" %->% phantom(0)),
    fill = "Model",
    linetype = NULL
  ) +
  guides(
    color = "none",
    fill = guide_legend(ncol = 1, override.aes = list(alpha = 1)),
    linetype = guide_legend(ncol = 1)
  ) +
  theme(
    axis.title.y = element_text(size = 7),
    plot.margin = margin(12, 52, 2, 2)
  )

########## Helpfulness / harmlessness frontier

# The labeled frontier models sit in one dense cluster, so their labels are
# stacked in a column to the right and joined to the points by leader lines
FRONTIER_LABEL_X <- 0.57
FRONTIER_LABEL_Y <- c(
  "gptoss20b"    = 0.97,
  "luna"         = 0.895,
  "sonnet5"      = 0.82,
  "gemini31pro"  = 0.745,
  "gemma27b"     = 0.67,
  "gemini3flash" = 0.595
)

# Average helpfulness and harmlessness of each frontier model, both rescaled
# to [0, 1] from worse to better: help is the 0-3 judge over 3, and safety is
# the 1-5 harm judge reverse-coded as (5 - harm) / 4
fleet_frontier <- fleet_judge %>%
  filter(eval == "harm", metric %in% c("help", "harm")) %>%
  mutate(
    value = case_when(
      metric == "harm" ~ (5 - value) / 4,
      metric == "help" ~ value / 3
    )
  ) %>%
  summarize_ci(model, metric) %>%
  pivot_wider(
    id_cols = model,
    names_from = metric,
    values_from = mean
  ) %>%
  mutate(
    label = TR_FLEET_FRONTIER[model],
    label_x = FRONTIER_LABEL_X,
    label_y = FRONTIER_LABEL_Y[model]
  )

fleet_frontier_labels <- fleet_frontier %>%
  drop_na(label)

# Average helpfulness and harmlessness of the policies trained on each help
# weight, rescaled the same way. beta_help = ∞ is the lexicographic help-first
# reward, the helpful end of the frontier.
steering_frontier <- policy_judge %>%
  filter(
    reward == reward_regime,
    eval == "harm",
    metric %in% c("help", "harm"),
    beta_lean == 0
  ) %>%
  mutate(
    value = case_when(
      metric == "harm" ~ (5 - value) / 4,
      metric == "help" ~ value / 3
    )
  ) %>%
  summarize_ci(model, beta_help, split, metric) %>%
  mutate(
    model = relabel(TR_MODEL)(model),
    split = relabel(TR_SPLIT)(split)
  ) %>%
  pivot_wider(
    id_cols = c(model, beta_help, split),
    names_from = metric,
    values_from = c(mean, std.err)
  )

# Plot the frontier traced by the help weight, with the frontier models as
# labeled points
p_frontier <- steering_frontier %>%
  ggplot(aes(
    x = mean_help,
    y = mean_harm,
    color = model,
    group = interaction(model, split)
  )) +
  geom_point(
    aes(x = help, y = harm),
    data = fleet_frontier,
    inherit.aes = FALSE,
    color = "grey50",
    size = 1.2
  ) +
  geom_segment(
    aes(x = help, y = harm, xend = label_x - 0.01, yend = label_y),
    data = fleet_frontier_labels,
    inherit.aes = FALSE,
    color = "grey70",
    linewidth = 0.25
  ) +
  geom_text(
    aes(x = label_x, y = label_y, label = label),
    data = fleet_frontier_labels,
    inherit.aes = FALSE,
    hjust = 0,
    color = "grey30",
    size = 1.6
  ) +
  geom_line(aes(linetype = split)) +
  geom_point() +
  scale_x_continuous(
    limits = c(0, 0.87),
    breaks = c(0, 0.25, 0.5, 0.75),
    labels = label_percent(),
    expand = expansion(mult = c(0.04, 0))
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    labels = label_percent(),
    expand = expansion(mult = c(0, 0.04)),
    oob = oob_keep
  ) +
  scale_color_manual(values = MODEL_COLORS) +
  coord_cartesian(clip = "off") +
  labs(
    x = expr(phantom(0) %<-% "Unhelpful" ~~ "Helpful" %->% phantom(0)),
    y = expr(phantom(0) %<-% "Harmful" ~~~ "Safe" %->% phantom(0))
  ) +
  guides(color = "none", linetype = "none") +
  theme(
    axis.title = element_text(size = 7),
    plot.margin = margin(12, 10, 2, 2)
  )

########## Combine the two panels with one shared legend

p_steering <- (p_slant | p_frontier) +
  plot_layout(widths = c(1, 1.2), guides = "collect") &
  theme(
    legend.box.just = "left",
    legend.title = element_text(size = 7),
    legend.text = element_text(size = 7),
    legend.key.size = unit(0.55, "lines"),
    legend.spacing.y = unit(2, "pt"),
    legend.box.spacing = unit(2, "pt")
  )

ggsave(
  path("figures", glue("steering{file_suffix}.pdf")),
  plot = p_steering,
  width = TEXT_WIDTH,
  height = 2.1,
  device = cairo_pdf
)

write_csv(
  steering_slant,
  path("figures", glue("steering_slant{file_suffix}_data.csv"))
)
write_csv(
  fleet_slant,
  path("figures", glue("steering_fleet_slant{file_suffix}_data.csv"))
)
write_csv(
  steering_frontier,
  path("figures", glue("steering_frontier{file_suffix}_data.csv"))
)
write_csv(
  fleet_frontier,
  path("figures", glue("steering_fleet_frontier{file_suffix}_data.csv"))
)

################################################################################
# Non-interference

# Each panel puts a metric against the knob that does not own it, one curve
# per level of the knob that does. Safety is the 1-5 harm judge reverse-coded
# to [0, 1]. The lexicographic arms sit past the finite grid (beta_help = ∞ at
# 1.4, beta_lean = ±∞ at ±5); the two doubly-infinite corners are not trained.
non_interference <- policy_judge %>%
  filter(
    reward == reward_regime,
    arm_type == "grid",
    metric %in% c("political_slant", "harm")
  ) %>%
  mutate(value = if_else(metric == "harm", (5 - value) / 4, value)) %>%
  summarize_ci(model, beta_help, beta_lean, split, metric) %>%
  mutate(
    model = relabel(TR_MODEL)(model),
    split = relabel(TR_SPLIT)(split),
    x_help = if_else(is.infinite(beta_help), 1.4, beta_help),
    x_lean = pmax(pmin(beta_lean, 5), -5),
    beta_help_f = factor(
      beta_help,
      levels = c(0, 0.25, 0.5, 1, Inf),
      labels = c("0", "0.25", "0.5", "1", "∞")
    ),
    beta_lean_f = factor(
      beta_lean,
      levels = c(-Inf, -3, -1, 0, 1, 3, Inf),
      labels = c("−∞", "−3", "−1", "0", "1", "3", "∞")
    )
  )

# Plot the slant against the help weight
# NOTE: The in/out-of-distribution guide is drawn once, from the lower panel,
#       so that it sorts last in the shared legend.
p_non_interference_slant <- non_interference %>%
  filter(metric == "political_slant") %>%
  ggplot(aes(
    x = x_help,
    y = mean,
    color = beta_lean_f,
    group = interaction(beta_lean_f, split)
  )) +
  geom_ribbon(
    aes(ymin = lo, ymax = hi, fill = beta_lean_f),
    color = NA,
    alpha = 0.2
  ) +
  geom_line(aes(linetype = split)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = 1.2, linetype = "dotted") +
  scale_x_continuous(
    breaks = c(0, 0.5, 1, 1.2, 1.4),
    labels = c("0", "0.5", "1", "···", "∞"),
    minor_breaks = c(0.25, 0.75)
  ) +
  scale_color_manual(values = LEAN_COLORS, aesthetics = c("color", "fill")) +
  scale_linetype_discrete(guide = "none") +
  labs(
    x = expr(beta["help"]),
    y = expr(phantom(0) %<-% "Liberal" ~~~ "Conservative" %->% phantom(0)),
    color = expr(beta["lean"]),
    fill = expr(beta["lean"])
  ) +
  guides(
    color = guide_legend(nrow = 1, order = 1),
    fill = guide_legend(nrow = 1, order = 1)
  ) +
  facet_wrap(vars(model)) +
  theme(
    panel.spacing = unit(1, "lines"),
    axis.title = element_text(size = 8),
    axis.text = element_text(size = 7),
    plot.margin = margin(9, 2, 2, 2)
  )

# Plot the safety against the lean weight
p_non_interference_harm <- non_interference %>%
  filter(metric == "harm") %>%
  ggplot(aes(
    x = x_lean,
    y = mean,
    color = beta_help_f,
    group = interaction(beta_help_f, split)
  )) +
  geom_ribbon(
    aes(ymin = lo, ymax = hi, fill = beta_help_f),
    color = NA,
    alpha = 0.2
  ) +
  geom_line(aes(linetype = split)) +
  geom_vline(xintercept = c(-4, 4), linetype = "dotted") +
  scale_x_continuous(
    breaks = c(-5, -4, -3, -1, 0, 1, 3, 4, 5),
    labels = c("−∞", "···", "−3", "−1", "0", "1", "3", "···", "∞")
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    labels = label_percent(),
    oob = oob_keep
  ) +
  scale_color_manual(values = HELP_COLORS, aesthetics = c("color", "fill")) +
  scale_linetype_discrete(guide = guide_legend(order = 3)) +
  labs(
    x = expr(beta["lean"]),
    y = expr(phantom(0) %<-% "Harmful" ~~~ "Safe" %->% phantom(0)),
    color = expr(beta["help"]),
    fill = expr(beta["help"]),
    linetype = NULL
  ) +
  guides(
    color = guide_legend(nrow = 1, order = 2),
    fill = guide_legend(nrow = 1, order = 2)
  ) +
  facet_wrap(vars(model)) +
  theme(
    panel.spacing = unit(1, "lines"),
    axis.title = element_text(size = 8),
    axis.text = element_text(size = 7),
    plot.margin = margin(9, 2, 2, 2)
  )

# Stack the two panels with one shared legend
p_non_interference <- (p_non_interference_slant / p_non_interference_harm) +
  plot_layout(guides = "collect") +
  plot_annotation(theme = theme(plot.margin = margin(3, 2, 2, 2))) &
  theme(
    legend.position = "bottom",
    legend.box = "vertical"
  )

ggsave(
  path("figures", glue("non_interference{file_suffix}.pdf")),
  plot = p_non_interference,
  width = TEXT_WIDTH,
  height = 5.9,
  device = cairo_pdf
)

write_csv(
  non_interference,
  path("figures", glue("non_interference{file_suffix}_data.csv"))
)

################################################################################
# Discrimination

# In distribution, a hiring cell pairs a female-name and a male-name prompt
# over the same two jobs, and alignment is the share of the two picks that
# follow the gender stereotype. Only cells where both picks parsed are kept.
discrimination_jobs <- hiring_picks %>%
  drop_na(aligned) %>%
  group_by(model, arm, rm, q, cell) %>%
  filter(n() == 2) %>%
  summarize(value = mean(aligned), .groups = "drop") %>%
  summarize_ci(q, rm, model) %>%
  mutate(split = "eval_in")

# Out of distribution, alignment is already per name pair, in {0, 0.5, 1}
discrimination_movies <- movie_picks %>%
  drop_na(congruence) %>%
  rename(value = congruence) %>%
  summarize_ci(q, rm, model) %>%
  mutate(split = "eval_out")

# The untrained base (no reward model, no dose) is left off
discrimination <- bind_rows(discrimination_jobs, discrimination_movies) %>%
  drop_na(rm) %>%
  mutate(
    model = relabel(TR_MODEL)(model),
    split = relabel(TR_SPLIT)(split),
    rm_class = if_else(
      rm %in% c("rubric", "rubric_constrained"),
      "Rubric RM",
      "Black-box RM"
    ),
    rm_class = factor(rm_class, levels = c("Black-box RM", "Rubric RM")),
    rm = relabel(TR_RM)(rm)
  )

# Plot the dose curves of the given models, with the neural and rubric reward
# models side by side for one model and one column per model for several
plot_discrimination <- function(df, models, file) {
  multi <- length(models) > 1
  facet_layer <- if (multi) {
    facet_grid(rows = vars(rm_class), cols = vars(model))
  } else {
    facet_wrap(vars(rm_class), nrow = 1)
  }
  height <- if (multi) 3 else 1.6
  y_title <- if (multi) "Evaluation gender alignment" else "Gender alignment"
  legend_columns <- if (multi) 1 else 2

  p_discrimination <- df %>%
    filter(model %in% TR_MODEL[models]) %>%
    ggplot(aes(
      x = q,
      y = mean,
      color = rm,
      group = interaction(rm, split)
    )) +
    geom_ribbon(
      aes(ymin = lo, ymax = hi, fill = rm),
      color = NA,
      alpha = 0.2
    ) +
    geom_line(aes(linetype = split)) +
    geom_hline(yintercept = 0.5, linetype = "dotted") +
    scale_x_continuous(
      breaks = c(0, 0.5, 1),
      labels = label_percent(),
      expand = c(0, 0),
      oob = oob_keep
    ) +
    scale_y_continuous(
      limits = c(0.4, 1),
      breaks = c(0.5, 0.75, 1),
      labels = label_percent(),
      expand = c(0, 0),
      oob = oob_keep
    ) +
    labs(
      x = expr("Injected bias" ~ q),
      y = y_title,
      color = "Reward model",
      fill = "Reward model",
      linetype = NULL
    ) +
    guides(
      color = guide_legend(order = 1, ncol = legend_columns),
      fill = guide_legend(order = 1, ncol = legend_columns),
      linetype = guide_legend(order = 2)
    ) +
    facet_layer +
    theme(
      axis.title = element_text(size = 8),
      axis.text = element_text(size = 7),
      legend.title = element_text(size = 7),
      legend.text = element_text(size = 7),
      legend.key.size = unit(0.55, "lines"),
      legend.spacing.y = unit(1, "pt"),
      legend.box.spacing = unit(2, "pt"),
      legend.box.just = "left",
      panel.spacing = unit(1.5, "lines")
    )

  ggsave(
    path("figures", file),
    plot = p_discrimination,
    width = TEXT_WIDTH,
    height = height,
    device = cairo_pdf
  )
}

plot_discrimination(
  discrimination,
  models = "gptoss",
  file = "discrimination_gptoss.pdf"
)

write_csv(discrimination, path("figures", "discrimination_data.csv"))

################################################################################
# Sycophancy

# Labels are pushed down where the frontier models sit closer together than
# one line of text
SYCOPHANCY_LABEL_GAP <- 0.17

# Average framing gap of each frontier model
fleet_sycophancy <- fleet_sycophancy_obs %>%
  drop_na(syc_framing_gap) %>%
  rename(value = syc_framing_gap) %>%
  summarize_ci(model) %>%
  mutate(label = TR_FLEET_SYCOPHANCY[model])

# Stack the labels of the labeled frontier models from the top down
fleet_sycophancy_labels <- fleet_sycophancy %>%
  drop_na(label) %>%
  arrange(desc(mean)) %>%
  mutate(
    label_y = accumulate(
      mean,
      \(above, current) min(current, above - SYCOPHANCY_LABEL_GAP)
    )
  )

# Average framing gap of the policies trained on each sycophancy weight. The
# lexicographic arm (beta_syc = ∞) sits at 4, past a break at 3.
sycophancy <- policy_sycophancy %>%
  drop_na(syc_framing_gap) %>%
  filter(reward == reward_regime) %>%
  rename(value = syc_framing_gap) %>%
  summarize_ci(model, split, beta_syc) %>%
  mutate(
    model = relabel(TR_MODEL)(model),
    split = relabel(TR_SPLIT)(split),
    x = pmin(beta_syc, 4)
  )

# Plot the framing gap against the sycophancy weight, with the frontier models
# as ticks in the right margin
p_sycophancy <- sycophancy %>%
  ggplot(aes(
    x = x,
    y = mean,
    color = model,
    group = interaction(model, split)
  )) +
  geom_ribbon(
    aes(ymin = lo, ymax = hi, fill = model),
    color = NA,
    alpha = 0.2
  ) +
  geom_line(aes(linetype = split)) +
  geom_vline(xintercept = 3, linetype = "dotted") +
  geom_segment(
    aes(y = mean, yend = mean),
    data = fleet_sycophancy,
    inherit.aes = FALSE,
    x = 4.02,
    xend = 4.24,
    color = "grey40",
    linewidth = 0.5
  ) +
  geom_segment(
    aes(y = mean, yend = label_y),
    data = fleet_sycophancy_labels,
    inherit.aes = FALSE,
    x = 4.24,
    xend = 4.3,
    color = "grey40",
    linewidth = 0.25
  ) +
  geom_text(
    aes(y = label_y, label = label),
    data = fleet_sycophancy_labels,
    inherit.aes = FALSE,
    x = 4.32,
    hjust = 0,
    color = "grey30",
    size = 1.8
  ) +
  scale_x_continuous(
    limits = c(0, 4),
    oob = oob_keep,
    breaks = c(0, 0.5, 1, 2, 3, 4),
    labels = c("0", "0.5", "1", "2", "···", "∞"),
    expand = c(0, 0)
  ) +
  scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.1))) +
  scale_color_manual(values = MODEL_COLORS, aesthetics = c("color", "fill")) +
  coord_cartesian(clip = "off") +
  labs(
    x = expr(beta["sycophancy"]),
    y = "Framing gap",
    color = "Model",
    fill = "Model",
    linetype = NULL
  ) +
  theme(
    axis.title = element_text(size = 8),
    axis.text = element_text(size = 7),
    legend.title = element_text(size = 7),
    legend.text = element_text(size = 7),
    legend.key.size = unit(0.6, "lines"),
    # The legend sits to the right of the frontier-model labels in the margin
    legend.box.spacing = unit(70, "pt"),
    legend.background = element_blank(),
    legend.key = element_blank(),
    plot.margin = margin(3, 2, 2, 2)
  )

ggsave(
  path("figures", glue("sycophancy{file_suffix}.pdf")),
  plot = p_sycophancy,
  width = 0.7 * TEXT_WIDTH,
  height = 1.8,
  device = cairo_pdf
)

write_csv(
  sycophancy,
  path("figures", glue("sycophancy{file_suffix}_data.csv"))
)
write_csv(
  fleet_sycophancy,
  path("figures", glue("sycophancy_fleet{file_suffix}_data.csv"))
)

################################################################################
################################ APPENDIX PLOTS ################################
################################################################################

################################################################################
# The rubric, shaded by the share of each subsection the fit kept

# Plot the rubric again, with the share of each subsection that the fit kept
# drawn as a darker band rising from the bottom of the subsection row
plot_rubric(
  rubric_structure,
  shaded = TRUE,
  width = TEXT_WIDTH,
  file = "rubric_shaded.pdf"
)

################################################################################
# Reward-model agreement on the other evaluations

plot_rm_correlation(
  rm_correlation,
  evals = c("feedback", "political", "jobs"),
  file = glue("rm_correlation_other{file_suffix}.pdf")
)

################################################################################
# Discrimination for every model

plot_discrimination(
  discrimination,
  models = c("gemma", "qwen", "gptoss"),
  file = "discrimination.pdf"
)

################################################################################
# Capability retention

# The steering-grid and sycophancy policies, one panel each, in the order of
# their weights
capability_arms <- capability %>%
  filter(arm_type %in% c("grid", "syc")) %>%
  arrange(family, beta_help, beta_lean, beta_syc) %>%
  mutate(label = fct_rev(fct_inorder(label)))

# The untrained base every policy is compared with, drawn in every panel with
# its 95% interval
capability_base <- capability %>%
  filter(arm_type == "base") %>%
  select(model, metric, value, lo, hi)

# The dose policies, one panel per reward model, by injected bias
capability_dose <- capability %>%
  filter(arm_type == "dose") %>%
  mutate(
    family = relabel(TR_RM)(rm),
    label = factor(
      q,
      levels = c(1, 0.8, 0.6, 0.4, 0.2, 0),
      labels = c("1", "0.8", "0.6", "0.4", "0.2", "0")
    )
  )

# The y-axis titles of the two families of capability plots
ARMS_AXIS_TITLE <- expression(
  "Trained weights   " * beta[help] * " / " * beta[lean] *
    "   and   " * beta[syc]
)
DOSE_AXIS_TITLE <-
  "Injected bias  q  (share of non-aligned preferences flipped)"

# Plot the accuracy of every policy in a frame on one benchmark against the
# untrained base, one row of panels per policy family and one column per model
plot_capability <- function(
    df,
    df_base,
    metric,
    y_label,
    height,
    file
) {
  p_capability <- df %>%
    filter(metric == .env$metric) %>%
    ggplot(aes(x = value, y = label)) +
    geom_rect(
      aes(xmin = lo, xmax = hi),
      data = filter(df_base, metric == .env$metric),
      inherit.aes = FALSE,
      ymin = -Inf,
      ymax = Inf,
      fill = "grey80",
      alpha = 0.5
    ) +
    geom_vline(
      aes(xintercept = value),
      data = filter(df_base, metric == .env$metric),
      linetype = "dashed",
      color = "grey35"
    ) +
    geom_errorbar(
      aes(xmin = lo, xmax = hi),
      width = 0,
      orientation = "y",
      color = "grey55",
      linewidth = 0.3
    ) +
    geom_point(size = 0.8, color = "#2b5581") +
    scale_x_continuous(
      labels = label_percent(accuracy = 1),
      breaks = seq(0.5, 0.9, by = 0.1)
    ) +
    labs(x = metric, y = y_label) +
    facet_grid(
      rows = vars(family),
      cols = vars(model),
      scales = "free_y",
      space = "free_y"
    ) +
    theme(
      axis.text.y = element_text(size = 6),
      axis.title.y = element_text(size = 8),
      panel.grid.major.y = element_line(color = "grey93"),
      strip.text.y.right = element_text(size = 6.5, angle = 0, hjust = 0),
      panel.spacing.x = unit(6, "pt"),
      plot.margin = margin(2, 4, 2, 2)
    )

  ggsave(
    path("figures", file),
    plot = p_capability,
    width = TEXT_WIDTH,
    height = height,
    device = cairo_pdf
  )
}

plot_capability(
  df = capability_arms,
  df_base = capability_base,
  metric = "MMLU",
  y_label = ARMS_AXIS_TITLE,
  height = 4.4,
  file = glue("capability_mmlu{file_suffix}.pdf")
)
plot_capability(
  df = capability_arms,
  df_base = capability_base,
  metric = "IFEval (strict)",
  y_label = ARMS_AXIS_TITLE,
  height = 4.4,
  file = glue("capability_ifeval{file_suffix}.pdf")
)
plot_capability(
  df = capability_dose,
  df_base = capability_base,
  metric = "MMLU",
  y_label = DOSE_AXIS_TITLE,
  height = 5,
  file = "capability_disc_mmlu.pdf"
)
plot_capability(
  df = capability_dose,
  df_base = capability_base,
  metric = "IFEval (strict)",
  y_label = DOSE_AXIS_TITLE,
  height = 5,
  file = "capability_disc_ifeval.pdf"
)

write_csv(
  capability,
  path("figures", glue("capability{file_suffix}_data.csv"))
)
