
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_DIR := $(CURRENT_DIR)/..

task ?= ojin

results_file := $(PROJECT_DIR)/get_results_file/output/$(task)/all_results.csv
bootstrap_results_file := $(PROJECT_DIR)/get_results_file/output/$(task)/all_bootstrapped_results.csv
config_file := $(CURRENT_DIR)/config.yaml

output_dir = $(CURRENT_DIR)/output/
output_file := $(output_dir)/significance_results.xlsx

$(output_file): get_significance_results.py $(config_file) $(results_file) $(bootstrap_results_file)
	mkdir -p $(output_dir)
	nice -n 10 python get_significance_results.py \
			--results_file=$(results_file) \
			--bootstrap_results_file=$(bootstrap_results_file) \
			--config_file=$(config_file) \
			--output_file=$(output_file)

.DEFAULT_GOAL = all

all: $(output_file)

clean: 
	rm -rf $(output_dir)
