CURRENT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

raw_data_file := /projects/2017-007-namematch/plosOne/raw_data/OJIN_Combined.csv

output_dir = $(CURRENT_DIR)/output
output_file = $(output_dir)/OJIN_for_paper.csv

$(output_file): $(raw_data_file) clean_ojin.R
	mkdir -p $(output_dir)
	Rscript clean_ojin.R \
		--raw_data_file=$(raw_data_file) \
		--output_file=$(output_file)
		
clean: 
	rm -rf $(output_dir)

