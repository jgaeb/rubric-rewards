library(fs)
library(glue)
library(scales)
library(nanoparquet)
library(argparse)
library(tidyverse)

# Parse command line arguments
parser <- ArgumentParser()
parser$add_argument(
  "--reward",
  type = "character",
  default = "free",
  choices = c("free", "constrained")
)
args <- parser$parse_args()

# Define the reward regime the policies were trained under
reward_regime <- args$reward

# Files drawn under the constrained reward carry a suffix
file_suffix <- if (reward_regime == "free") "" else "_constrained"

# Set the seed for replicability
set.seed(31415926)

# Set the ggplot theme
theme_set(theme_bw(base_size = 10))
theme_update(
  legend.margin = margin(),
  legend.text   = element_text(size = 10),
  plot.margin   = margin(2, 2, 2, 2)
)

# Register the helper functions
source("utils.R")
