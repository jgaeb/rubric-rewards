# Set the Hugging Face dataset the package is downloaded from
REPO ?= jgaeb/rubric-rewards

.PHONY: all fetch tables refit figures check clean deep-clean
.DEFAULT_GOAL := all

# The multi-target rules below must not run in parallel
.NOTPARALLEL:

################################################################################
# Data

# The package is downloaded when the manifest is missing; remove the manifest
# to download it again
MANIFEST_raw.json:
	@echo "########################################"
	@echo "Downloading the package from $(REPO)..."
	python3 fetch.py \
		--repo $(REPO) \
		--what all

fetch: MANIFEST_raw.json

RAW_DIRECTORIES = raw/rubric/ \
                  raw/prompts/ \
                  raw/responses/ \
                  raw/grades/ \
                  raw/rankings/ \
                  raw/rewards/ \
                  raw/sft/ \
                  raw/policy/ \
                  raw/fleet/ \
                  raw/discrimination/ \
                  raw/rm_scores/

# The raw records come with the download
$(RAW_DIRECTORIES): | MANIFEST_raw.json

################################################################################
# Tables

TABLES = tables/rubric_items.parquet \
         tables/rubric_structure.parquet \
         tables/prompts_feedback.parquet \
         tables/prompts_harm.parquet \
         tables/prompts_political.parquet \
         tables/prompts_jobs.parquet \
         tables/prompts_movies.parquet \
         tables/prompts_mmlu.parquet \
         tables/prompts_ifeval.parquet \
         tables/responses.parquet \
         tables/grades.parquet \
         tables/rankings.parquet \
         tables/weights.parquet \
         tables/sft_targets.parquet \
         tables/policy_judge_obs.parquet \
         tables/policy_syc_obs.parquet \
         tables/policy_capability_obs.parquet \
         tables/policy_generations.parquet \
         tables/fleet_judge_obs.parquet \
         tables/fleet_syc_obs.parquet \
         tables/disc_jobs_obs.parquet \
         tables/disc_movies_obs.parquet \
         tables/rm_scores.parquet \
         tables/rm_correlations.parquet

$(TABLES): $(RAW_DIRECTORIES) \
           build_tables.py \
           _utils.py
	@echo "########################################"
	@echo "Building the tables from the raw records..."
	@date
	time python3 build_tables.py

tables: $(TABLES)

################################################################################
# Reward refits

REFITS = refit/gemma_free.json \
         refit/gemma_constrained.json \
         refit/qwen_free.json \
         refit/qwen_constrained.json \
         refit/gptoss_free.json \
         refit/gptoss_constrained.json

refit/%_free.json refit/%_constrained.json: tables/grades.parquet \
                                            tables/rankings.parquet \
                                            tables/weights.parquet \
                                            tables/sft_targets.parquet \
                                            tables/responses.parquet \
                                            raw/rubric/ \
                                            raw/rankings/ \
                                            raw/discrimination/ \
                                            refit.py \
                                            _utils.py
	@echo "########################################"
	@echo "Refitting the $* rewards and re-deriving the training picks..."
	@date
	time python3 refit.py \
		--model $* \
		--dose \
		--write refit

refit: $(REFITS)

################################################################################
# Figures

FIGURES = figures/rubric.pdf \
          figures/rubric_shaded.pdf \
          figures/rm_correlation.pdf \
          figures/rm_correlation_other.pdf \
          figures/steering.pdf \
          figures/non_interference.pdf \
          figures/discrimination.pdf \
          figures/discrimination_gptoss.pdf \
          figures/sycophancy.pdf \
          figures/capability_mmlu.pdf \
          figures/capability_ifeval.pdf \
          figures/capability_disc_mmlu.pdf \
          figures/capability_disc_ifeval.pdf \
          figures/rubric_data.csv \
          figures/rm_correlation_data.csv \
          figures/steering_slant_data.csv \
          figures/steering_fleet_slant_data.csv \
          figures/steering_frontier_data.csv \
          figures/steering_fleet_frontier_data.csv \
          figures/non_interference_data.csv \
          figures/discrimination_data.csv \
          figures/sycophancy_data.csv \
          figures/sycophancy_fleet_data.csv \
          figures/capability_data.csv

FIGURES_CONSTRAINED = figures/rm_correlation_constrained.pdf \
                      figures/rm_correlation_other_constrained.pdf \
                      figures/steering_constrained.pdf \
                      figures/non_interference_constrained.pdf \
                      figures/sycophancy_constrained.pdf \
                      figures/capability_mmlu_constrained.pdf \
                      figures/capability_ifeval_constrained.pdf \
                      figures/rm_correlation_constrained_data.csv \
                      figures/steering_slant_constrained_data.csv \
                      figures/steering_fleet_slant_constrained_data.csv \
                      figures/steering_frontier_constrained_data.csv \
                      figures/steering_fleet_frontier_constrained_data.csv \
                      figures/non_interference_constrained_data.csv \
                      figures/sycophancy_constrained_data.csv \
                      figures/sycophancy_fleet_constrained_data.csv \
                      figures/capability_constrained_data.csv

FIGURE_TABLES = tables/rubric_structure.parquet \
                tables/policy_judge_obs.parquet \
                tables/fleet_judge_obs.parquet \
                tables/disc_jobs_obs.parquet \
                tables/disc_movies_obs.parquet \
                tables/policy_syc_obs.parquet \
                tables/fleet_syc_obs.parquet \
                tables/rm_correlations.parquet \
                tables/policy_capability_obs.parquet

$(FIGURES): $(FIGURE_TABLES) \
            plots.R \
            setup.R \
            utils.R
	@echo "########################################"
	@echo "Generating the figures..."
	@date
	time Rscript plots.R

# The constrained pass also redraws the figures that do not depend on the
# reward, so it runs after the free pass
$(FIGURES_CONSTRAINED): $(FIGURE_TABLES) \
                        plots.R \
                        setup.R \
                        utils.R \
                        | $(FIGURES)
	@echo "########################################"
	@echo "Generating the figures for the constrained reward..."
	@date
	time Rscript plots.R \
		--reward constrained

figures: $(FIGURES) \
         $(FIGURES_CONSTRAINED)

################################################################################
# Acceptance test

check: check.py \
       build_tables.py \
       _utils.py \
       plots.R \
       setup.R \
       utils.R \
       | MANIFEST_raw.json
	@echo "########################################"
	@echo "Checking that the package reproduces from the raw records..."
	@date
	time python3 check.py \
		--manifest MANIFEST_raw.json

################################################################################
# Aggregate targets

all: $(TABLES) \
     $(FIGURES) \
     $(FIGURES_CONSTRAINED)

clean:
	@echo "Cleaning up"
	@rm -f figures/*.pdf
	@rm -f figures/*_data.csv
	@rm -f refit/*.json
	@rm -f *.log

deep-clean: clean
	@echo "Cleaning up the tables"
	@rm -f tables/*.parquet
