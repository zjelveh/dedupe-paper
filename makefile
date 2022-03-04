
all: 
	cd clean_ojin && make
	cd generate_datasets && make task=ojin
	cd run_dedupe && make task=ojin
	cd run_nm && make task=ojin
	cd get_results_file && make task=ojin
	cd create_figures && make task=ojin
