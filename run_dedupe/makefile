CURRENT_DIR:= $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

task ?= ojin

output_dir := $(CURRENT_DIR)/output/$(task)
exp_file := $(CURRENT_DIR)/dedupe_experiment.py
failed_runs_file := $(output_dir)/failed_runs.csv

ojin_ss_grid := 200000 50000 5000

dataset_version_list := low_br
blocked_proportion_list := 90 50
ss_train := $($(task)_ss_grid)
em_train := 50
budget := 20 40 80 200 500 1000
type := dedupe record_linkage
sample_size := 15000 150000 1500 300000
iteration := 0 1 2 3 4

all_recipes := 

define recipe_template = 
$(output_dir)/results__$(7)__sst_$(1)__emt_$(2)__budget_$(3)__sample_size_$(5)__block_prop_$(8)__type_$(4)__iter_$(6).csv:
	mkdir -p $(output_dir)
	python $(exp_file) --dataset_version=$(dataset_version) \
	--ss_train=$(1) --em_train=$(2) --budget=$(3) --type=$(4) --sample_size=$(5) --iteration=$(6) --blocked_proportion=$(8) \
	--dataset_dir=$(CURRENT_DIR)/../generate_datasets/all_output/$(7)/$(task) \
	--output_file=$(output_dir)/results__$(7)__sst_$(1)__emt_$(2)__budget_$(3)__sample_size_$(5)__block_prop_$(8)__type_$(4)__iter_$(6).csv \
	--output_dir=$(output_dir) \
	--failed_runs_file=$(failed_runs_file)
		
all_recipes+=$(output_dir)/results__$(7)__sst_$(1)__emt_$(2)__budget_$(3)__sample_size_$(5)__block_prop_$(8)__type_$(4)__iter_$(6).csv
endef

$(foreach emt,$(em_train),\
	$(foreach bp,$(blocked_proportion_list),\
		$(foreach t,$(type),\
			$(foreach i,$(iteration),\
				$(foreach ss,$(sample_size),\
					$(foreach sst,$(ss_train),\
						$(foreach b,$(budget),\
							$(foreach dataset_version,$(dataset_version_list),\
							$(eval $(call recipe_template,$(sst),$(emt),$(b),$(t),$(ss),$(i),$(dataset_version),$(bp)))))))))))

.DEFAULT_GOAL = all

all: $(all_recipes)