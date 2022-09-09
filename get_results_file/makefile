CURRENT_DIR:= $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

task ?= ojin

nm_result_dir_path := $(CURRENT_DIR)/../run_nm/output/$(task)
dedupe_result_dir_path := $(CURRENT_DIR)/../run_dedupe/output/$(task)
fl_result_dir_path := $(CURRENT_DIR)/../run_fastlink/output/$(task)
config_file := $(CURRENT_DIR)/config.yaml
output_dir := $(CURRENT_DIR)/output/$(task)
result_file := $(output_dir)/all_results.csv

results: get_results_file.py $(fl_result_dir_path)/results.csv $(nm_result_dir_path)/results.csv
	mkdir -p $(output_dir)
	python get_results_file.py \
		--namematch_result_dir_path=$(nm_result_dir_path) \
		--dedupe_result_dir_path=$(dedupe_result_dir_path) \
		--fastlink_result_dir_path=$(fl_result_dir_path) \
		--config_file=$(config_file) \
		--output_file=$(result_file)

clean:
	- rm -rf $(output_dir)

.DEFAULT_GOAL = results
