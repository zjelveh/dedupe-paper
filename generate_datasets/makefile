
task ?= ojin

CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_DIR := $(CURRENT_DIR)/..

ojin_data_file := $(PROJECT_DIR)/clean_ojin/output/OJIN_for_paper.csv

data_file := $($(task)_data_file)
config_file := $(CURRENT_DIR)/$(task)_config.yaml

output_dir = $(CURRENT_DIR)/output/$(task)
output_file := $(output_dir)/task.log

$(output_file): generate_datasets.py $(config_file) $(data_file)
	mkdir -p $(output_dir)
	nice -n 10 python generate_datasets.py \
			--data_file=$(data_file) \
			--config_file=$(config_file) \
			--output_dir=$(output_dir)

.DEFAULT_GOAL = all

all: $(output_file)

clean: 
	rm -rf $(output_dir)

