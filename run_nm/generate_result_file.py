import argparse
import pandas as pd
import numpy as np
import os
import re
import sys
import logging
import yaml

sys.path.append('../utils/')
import eval_utils


def hms_to_m(hms):

    try:
        t = 0
        for u in hms.split(':'):
            t = 60 * t + int(u)
        t = t/60
    except:
        # accounts for a very odd issue where for 200k runs, the data-rows elapsed 
        # time stat is already an int (that we THINK might be seconds?)
        t = t/60

    return t


def main(nm_result_dir, output_file):

    spec_list = [d for d in os.listdir(nm_result_dir) if 'spec__' in d]

    result_list = []
    bs_result_list = []
                                
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

        an_w_cluster_id = pd.concat([exp_eval, admin_eval])

        ###

        result_df_row = eval_utils.eval_predictions(
            an_w_cluster_id, ss_train=sst, em_train=emt, ss_eval=sse, em_eval=eme, framework='', budget=0, dedupe_sample_size=0, model_iter=i)
        bs_result_df = eval_utils.bootstrap_eval_predictions(
            an_w_cluster_id, ss_train=sst, em_train=emt, ss_eval=sse, em_eval=eme, framework='', budget=0, dedupe_sample_size=0, model_iter=i)

        runtime_results_dict = {
            k.replace('__main', '_min') : hms_to_m(v)
            for task, task_stats in log_yaml['stats'].items() 
            for k, v in task_stats.items() if 'runtime' in k
        }
        runtime_results_df_row = pd.DataFrame.from_records([runtime_results_dict])
        result_df_row = pd.concat([result_df_row, runtime_results_df_row], axis=1)
        
        result_list.append(result_df_row.copy())
        bs_result_list.append(bs_result_df.copy())

    results = pd.concat(result_list)
    bs_results = pd.concat(bs_result_list)
    
    results.to_csv(output_file, index=None)
    bs_results.to_csv(output_file.replace('result', 'bootstrapped_result'), index=None)
                    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nm_result_dir')
    parser.add_argument('--output_file')
    args = parser.parse_args()

    main(args.nm_result_dir,
         args.output_file)

