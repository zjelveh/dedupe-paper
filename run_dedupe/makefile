CURRENT_DIR:= $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

task ?= ojin

output_dir := $(CURRENT_DIR)/output/$(task)
source_file := $(CURRENT_DIR)/dedupe_experiment.py
failed_runs_file := $(output_dir)/failed_runs.csv

ojin_ss_grid := 5000 50000 200000

ss_train := $($(task)_ss_grid)
em_train := 50
budget := 20 40 80 200 500 1000
type := deduplication record_linkage
sample_size := 1500 15000 150000 300000
iteration := 0 1 2 3 4 5 6 7 8 9

all_recipes := 

define recipe_template = 
$(eval output_file := $(output_dir)/results__sst_$(1)__emt_$(2)__budget_$(3)__sample_size_$(5)__type_$(4)__iter_$(6).csv)
$(output_file):
	mkdir -p $(output_dir)
	python $(source_file) \
	--ss_train=$(1) --em_train=$(2) --budget=$(3) --type=$(4) --sample_size=$(5) --iteration=$(6) \
	--dataset_dir=$(CURRENT_DIR)/../generate_datasets/output/$(task) \
	--output_file=$(output_dir)/results__sst_$(1)__emt_$(2)__budget_$(3)__sample_size_$(5)__type_$(4)__iter_$(6).csv \
	--output_dir=$(output_dir) \
	--failed_runs_file=$(failed_runs_file)

all_recipes+=$(output_file)
endef

$(foreach i,$(iteration),\
	$(foreach emt,$(em_train),\
		$(foreach ss,$(sample_size),\
			$(foreach t,$(type),\
				$(foreach sst,$(ss_train),\
					$(foreach b,$(budget),\
						$(eval $(call recipe_template,$(sst),$(emt),$(b),$(t),$(ss),$(i)))))))))
							

.DEFAULT_GOAL = all

all: $(all_recipes)