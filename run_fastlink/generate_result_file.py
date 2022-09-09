import argparse
import pandas as pd
import numpy as np
import os
import re
import sys
import logging

sys.path.append('../utils/')
import eval_utils 
from cluster_utils import cluster_edges, convert_predictions_to_all_names_with_clusterid


def main(dataset_dir, result_dir, output_file):

    result_files = [d for d in os.listdir(result_dir) if 'predicted_matches_' in d]

    result_list = []
    bs_result_list = []
                                
    for spec_result_file in result_files:

        print(spec_result_file)
        
        sse = int(spec_result_file.split('__')[1].split('_')[1])
        eme = int(spec_result_file.split('__')[2].split('_')[1])
        the_type = spec_result_file.split('__')[3].replace('type_', '')
        i = int(spec_result_file.split('__')[4].split('_')[1].replace('.csv', ''))
        sst = sse
        emt = eme

        try:
            exp_eval = pd.read_csv(os.path.join(dataset_dir, f'experiment_evaluation__{sse}__{eme}.csv'), low_memory=False)
            admin_eval = pd.read_csv(os.path.join(dataset_dir, f'admin_evaluation__{sse}__{eme}.csv'), low_memory=False)
            predicted_matches = pd.read_csv(os.path.join(result_dir, spec_result_file))
        except:
            print(dataset_dir_to_use)
            print(sse)
            print(eme)
            continue
        
        an_w_cluster_id = convert_predictions_to_all_names_with_clusterid(exp_eval, admin_eval, predicted_matches=predicted_matches)

        result_df_row = eval_utils.eval_predictions(
            an_w_cluster_id, ss_train=sst, em_train=emt, ss_eval=sse, em_eval=eme, framework=the_type, budget=0, dedupe_sample_size=0, model_iter=i)
        bs_result_df = eval_utils.bootstrap_eval_predictions(
            an_w_cluster_id, ss_train=sst, em_train=emt, ss_eval=sse, em_eval=eme, framework=the_type, budget=0, dedupe_sample_size=0, model_iter=i)

        result_list.append(result_df_row.copy())
        bs_result_list.append(bs_result_df.copy())

    results = pd.concat(result_list)
    bs_result = pd.concat(bs_result_list)
    
    results.to_csv(output_file, index=None)
    bs_result.to_csv(output_file.replace('result', 'bootstrapped_result'), index=None)
                  

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir')
    parser.add_argument('--result_dir')
    parser.add_argument('--output_file')
    args = parser.parse_args()

    main(args.dataset_dir,
         args.result_dir,
         args.output_file)
