CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_DIR := $(CURRENT_DIR)/..

task ?= ojin

dataset_dir := $(PROJECT_DIR)/generate_datasets/all_output/low_br/$(task)
config_template_file := $(CURRENT_DIR)/config_template.yaml

ojin_n_grid := 5000 200000 50000
iter_grid := 0

n_grid := $($(task)_n_grid)
share_exact_grid := 50

output_dir = $(CURRENT_DIR)/output/$(task)
result_file := $(output_dir)/results.csv

# instantiate an empty list of targets
target_list :=

define prep_and_run_nm
$(eval specification_str := $1_$2_$3_$4_$5)
$(eval training_file := $(dataset_dir)/admin_training__$1__$2.csv)
$(eval exp_file := $(dataset_dir)/experiment_evaluation__$3__$4.csv)
$(eval evaluation_file := $(dataset_dir)/admin_evaluation__$3__$4.csv)
$(eval specification_output_dir := $(output_dir)/spec__$(specification_str))

$(eval specification_output := $(specification_output_dir)/output/experiment_with_clusterid.csv)
$(specification_output): generate_specification_dirs.py $(config_template_file) $(training_file) $(exp_file) $(evaluation_file)
	mkdir -p $(specification_output_dir)
	python generate_specification_dirs.py \
		--config_template_file=$(config_template_file) \
		--specification_dir=$(specification_output_dir) \
		--task=$(task) \
		--experiment_file=$(exp_file) \
		--training_file=$(training_file) \
		--evaluation_file=$(evaluation_file)
	-cd $(specification_output_dir) && namematch --config-file=config.yaml --output-dir=output --cluster-constraints-file=../../../constraints.py run
target_list += $(specification_output)
endef

$(foreach training_share_exact,$(share_exact_grid),\
	$(foreach iter,$(iter_grid),\
		$(foreach training_n,$(n_grid),\
			$(foreach eval_share_exact,$(share_exact_grid),\
				$(eval $(call prep_and_run_nm,$(training_n),$(training_share_exact),$(training_n),$(eval_share_exact),$(iter)))))))

$(result_file): generate_result_file.py $(output_dir)/spec__5000_50_5000_50_0/output/experiment_with_clusterid.csv
	python generate_result_file.py \
		--nm_result_dir=$(output_dir) \
		--output_file=$(result_file)


.DEFAULT_GOAL = all

all: $(target_list) $(result_file)

result_file: $(result_file)

clean:
	- rm -rf $(output_dir)


