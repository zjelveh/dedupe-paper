CURRENT_DIR:= $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

task ?= ojin

dataset_dir_path := $(CURRENT_DIR)/../generate_datasets/all_output/dataset_version/$(task)
nm_result_dir_path := $(CURRENT_DIR)/../run_nm/output/$(task)
dedupe_result_dir_path := $(CURRENT_DIR)/../run_dedupe/output/$(task)
config_file := $(CURRENT_DIR)/config.yaml
output_dir := $(CURRENT_DIR)/output/$(task)
result_file := $(output_dir)/all_results.csv

$(result_file): get_results_file.py
	mkdir -p $(output_dir)
	python get_results_file.py \
		--dataset_dir_path=$(dataset_dir_path) \
		--nm_result_dir_path=$(nm_result_dir_path) \
		--dedupe_result_dir_path=$(dedupe_result_dir_path) \
		--config_file=$(config_file) \
		--output_file=$(result_file)

clean:
	- rm -rf $(output_dir)
