CURRENT_DIR:= $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

task ?= ojin

dataset_dir_path := $(CURRENT_DIR)/../generate_datasets/output/$(task)
config_file := $(CURRENT_DIR)/config.yaml
output_dir := $(CURRENT_DIR)/output/$(task)
result_file := $(output_dir)/results.csv

ojin_ss_grid := 5000 50000 200000

ss_eval := $($(task)_ss_grid)
em_eval := 50
type := deduplication record_linkage
iteration := 0 1 2 3 4 5 6 7 8 9

all_recipes := 

define recipe_template =
$(eval dataset_dir := $(CURRENT_DIR)/../generate_datasets/output/$(task))
$(eval input_file := $(dataset_dir)/admin_evaluation__$(1)__$(2).csv)
$(eval output_file := $(output_dir)/predicted_matches__sst_$(1)__emt_$(2)__type_$(3)__iter_$(4).csv)
$(output_file): fastlink.R $(input_file) $(config_file)
	mkdir -p $(output_dir)
	-/opt/R/3.6.2/bin/Rscript fastlink.R \
	--dataset_dir=$(dataset_dir) \
	--config_file=$(config_file) \
	--ss=$(1) --em=$(2) --type=$(3) --iter=$(4) --output_dir=$(output_dir)
all_recipes+=$(output_file)
endef

$(foreach i,$(iteration),\
	$(foreach t,$(type),\
		$(foreach eme,$(em_eval),\
			$(foreach sse,$(ss_eval),\
				$(eval $(call recipe_template,$(sse),$(eme),$(t),$(i)))))))


.DEFAULT_GOAL = links

links: $(all_recipes)

result_file: generate_result_file.py
	python generate_result_file.py \
		--dataset_dir=$(dataset_dir_path) \
		--result_dir=$(output_dir) \
		--output_file=$(result_file)

clean:
	- rm -rf $(output_dir)
