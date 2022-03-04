import argparse
import pandas as pd
import numpy as np
import os
import re
import sys
import logging
import yaml
from datetime import timedelta

sys.path.append('../utils/')
import eval_utils

def main(nm_result_dir, output_file):

    spec_list = [d for d in os.listdir(nm_result_dir) if 'spec__' in d]

    result_rows = []
                                
    for spec in spec_list:
        
        sst = int(spec.split('__')[1].split('_')[0])
        emt = int(spec.split('__')[1].split('_')[1])
        sse = int(spec.split('__')[1].split('_')[2])
        eme = int(spec.split('__')[1].split('_')[3])
        i = int(spec.split('__')[1].split('_')[4])

        try:
            exp_eval = pd.read_csv(os.path.join(nm_result_dir, f'{spec}/output/experiment_with_clusterid.csv'), low_memory=False)
            admin_eval = pd.read_csv(os.path.join(nm_result_dir, f'{spec}/output/admin_evaluation_with_clusterid.csv'), low_memory=False)
            log_yaml = yaml.load(open(os.path.join(nm_result_dir, f'{spec}/output/details/nm_info.yaml'), 'r'), Loader=yaml.FullLoader)
        except:
            continue

        both = pd.concat([exp_eval, admin_eval])

        evals = eval_utils.eval_predictions(both)

        results_dict = {
            'ss_train': sst,
            'em_train': emt,
            'ss_eval': sse,
            'em_eval': eme,
            'iter': i,
            'tpr_any': evals.tpr_any.iloc[0],
            'prc_any': evals.prc_any.iloc[0],
            'tnr_any': evals.tnr_any.iloc[0],
            'error_any': evals.error_any.iloc[0],
            'acc_any': evals.acc_any.iloc[0],
            'tpr_corr': evals.tpr_corr.iloc[0],
            'prc_corr': evals.prc_corr.iloc[0],
            'tpr_pair': evals.tpr_pair.iloc[0],
            'prc_pair': evals.prc_pair.iloc[0],
            'tnr_pair': evals.tnr_pair.iloc[0],
            'error_pair': evals.error_pair.iloc[0]
        }
        runtime_results_dict = {
            k.replace('__main', '_min') : v
            for task, task_stats in log_yaml['stats'].items() 
            for k, v in task_stats.items() if 'runtime' in k
        }
        results_dict.update(runtime_results_dict)
        result_rows.append(results_dict.copy())

    results = pd.DataFrame.from_records(result_rows)[list(result_rows[-1].keys())]
    
    results.to_csv(output_file, index=None)
        
                    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nm_result_dir')
    parser.add_argument('--output_file')
    args = parser.parse_args()

    main(args.nm_result_dir,
         args.output_file)

