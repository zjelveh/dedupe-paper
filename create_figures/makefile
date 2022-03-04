
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_DIR := $(CURRENT_DIR)/..

task ?= ojin

results_file := $(PROJECT_DIR)/get_results_file/output/$(task)/all_results.csv
nm_results_dir := $(PROJECT_DIR)/run_nm/output/$(task)/
config_file := $(CURRENT_DIR)/config.yaml

output_dir = $(CURRENT_DIR)/output/
output_file := $(output_dir)/task.log

$(output_file): create_figures.py plot_functions.py $(config_file) $(results_file)
	mkdir -p $(output_dir)
	nice -n 10 python create_figures.py \
			--results_file=$(results_file) \
			--nm_results_dir=$(nm_results_dir) \
			--config_file=$(config_file) \
			--output_dir=$(output_dir)

.DEFAULT_GOAL = all

all: $(output_file)

clean: 
	rm -rf $(output_dir)
