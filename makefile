
# in R
setup:
	cd clean_ojin && make

# in python (activate the dedupe_paper_py conda environment)
run_experiments:
	cd generate_datasets && make task=ojin
	cd run_dedupe && make task=ojin
	cd run_nm && make task=ojin

# in R
fastlink: 
	cd run_fastlink && make links task=ojin

# in python (activate the dedupe_paper_py conda environment)
results: 
	cd run_fastlink && make result_file task=ojin
	cd get_results_file && make task=ojin
	cd create_figures && make task=ojin
	cd get_significance_results && make task=ojin
