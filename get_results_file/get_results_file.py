import pandas as pd
import numpy as np
import argparse
import yaml
import os

import sys
sys.path.append('/projects/2017-007-namematch/plosOne/melissa_work/repo/utils/')
from eval_utils import eval_predictions
from cluster_utils import cluster_edges


def concat_all_result_files(nm_dir, dedupe_dir, fl_dir, bootstrapped=False):

    if not bootstrapped:
        dedupe_results_files = [os.path.join(dedupe_dir, f) for f in os.listdir(dedupe_dir) if f.startswith('result')]
        nm_results_files = [os.path.join(nm_dir, 'results.csv')]
        fl_results_files = [os.path.join(fl_dir, f) for f in os.listdir(fl_dir) if f.startswith('result')]
    else:
        dedupe_results_files = [os.path.join(dedupe_dir, f) for f in os.listdir(dedupe_dir) if f.startswith('bootstrapped_result')]
        nm_results_files = [os.path.join(nm_dir, 'bootstrapped_results.csv')]
        fl_results_files = [os.path.join(fl_dir, f) for f in os.listdir(fl_dir) if f.startswith('bootstrapped_result')]

    dedupe_results = pd.concat([pd.read_csv(r) for r in dedupe_results_files])
    dedupe_results['algorithm'] = 'dedupe' + ' (' + dedupe_results.framework + ')'

    nm_results = pd.concat([pd.read_csv(r) for r in nm_results_files])
    nm_results['algorithm'] = 'namematch'

    fl_results = pd.concat([pd.read_csv(r) for r in fl_results_files])
    fl_results['algorithm'] = 'fastlink' + ' (' + fl_results.framework + ')'
    
    results = pd.concat([dedupe_results, nm_results, fl_results], sort=False, join='outer')

    results['f1_any'] = (2 * (results.prc_any * results.tpr_any)) / (results.prc_any + results.tpr_any)
    if not bootstrapped:
        results['f1_corr'] = (2 * (results.prc_corr * results.tpr_corr)) / (results.prc_corr + results.tpr_corr)
        results['runtime_min'] = results.filter(regex='runtime_').sum(axis=1)
    
    sort_order = ['em_train', 'em_eval', 'ss_train', 'ss_eval', 'budget', 'model_iter', 'algorithm', 'dedupe_sample_size', 'budget']
    if bootstrapped:
        sort_order = sort_order + ['bs_iter']
    results = results.sort_values(sort_order)
    
    return results


def main(nm_dir, dedupe_dir, fl_dir, params, output_file): 

    results = concat_all_result_files(nm_dir, dedupe_dir, fl_dir, bootstrapped=False)
    bs_results = concat_all_result_files(nm_dir, dedupe_dir, fl_dir, bootstrapped=True)

    results = results[results.model_iter <= (params['n_model_runs_to_include'] - 1)]
    bs_results = bs_results[bs_results.model_iter <= (params['n_model_runs_to_include'] - 1)]

    results.to_csv(output_file, index=False)
    bs_results.to_csv(output_file.replace('all_results', 'all_bootstrapped_results'), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--namematch_result_dir_path')
    parser.add_argument('--dedupe_result_dir_path')
    parser.add_argument('--fastlink_result_dir_path')
    parser.add_argument('--config_file')
    parser.add_argument('--output_file')
    args = parser.parse_args()

    params = yaml.safe_load(open(args.config_file,'r'))

    main(args.namematch_result_dir_path,
         args.dedupe_result_dir_path,
         args.fastlink_result_dir_path,
         params,
         args.output_file)
