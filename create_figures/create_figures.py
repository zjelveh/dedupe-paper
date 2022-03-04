
import pandas as pd
import os
import yaml
import numpy as np
import argparse
import logging

import sys
sys.path.append('../utils/')
import eval_utils as eval_utils

from plot_functions import get_main_result_plots, get_sample_size_plots

def get_name_match_bootstrap_results(nm_results_dir, n_bs, n_admin_rows_list):

    logging.info("Bootstrapping name match result")

    nm_bs_results = []
    for ss in n_admin_rows_list:
        for i in [0]:
            spec = os.path.join(nm_results_dir, f'spec__{ss}_50_{ss}_50_{i}/')
            try:
                exp_eval = pd.read_csv(f'{spec}/output/experiment_with_clusterid.csv', low_memory=False)
                admin_eval = pd.read_csv(f'{spec}/output/admin_evaluation_with_clusterid.csv', low_memory=False)
            except: 
                continue
            
            for bs_iter in np.arange(n_bs):

                if bs_iter % 100 == 0:
                    logging.info(f"On bs iter {bs_iter}")

                both = pd.concat([
                    exp_eval.sample(frac=1, replace=True), 
                    admin_eval.sample(frac=1, replace=True)
                ])

                evals = eval_utils.eval_predictions(both)
                evals['ss_train'] = ss
                evals['iter'] = i
                evals['bs_iter'] = bs_iter          
                nm_bs_results.append(evals)
                
    nm_bs_results = pd.concat(nm_bs_results)

    metrics = nm_bs_results.filter(regex='tpr|prc|error').columns.tolist()
    nm_bs_results = nm_bs_results.groupby(['ss_train', 'iter'])[metrics].quantile([.025, .5, .975]).unstack().stack(level=0)
    nm_bs_results.columns = ['lower_bound', 'median', 'upper_bound']
    nm_bs_results = nm_bs_results.unstack()
    nm_bs_results.columns = ['__'.join(col).strip() for col in nm_bs_results.columns.values]
    nm_bs_results = nm_bs_results.reset_index()

    nm_bs_results['algorithm'] = 'namematch'
    nm_bs_results['ss_eval'] = nm_bs_results.ss_train
    nm_bs_results['em_train'] = 50
    nm_bs_results['em_eval'] = 50

    return nm_bs_results


def create_figure(all_results, plot_name, plot_params, figure_output_dir):

    logging.info(f"Creating the {plot_name} plot...")

    if plot_params['function'] == 'get_main_result_plots':

        if plot_params['dedupe_defaults']:
            results = all_results[
                (all_results.algorithm == 'namematch') | 
                ((all_results['type'] == 'dedupe') & (all_results.blocked_proportion == 90) & (all_results.sample_size == 1500)) | 
                ((all_results['type'] == 'record_linkage') & (all_results.blocked_proportion == 50) & (all_results.sample_size == 15000))
            ].copy()
            force_ylim = 'default'
        else:
            results = all_results[
                (all_results.sample_size.isin([150000, 0])) & 
                (all_results.blocked_proportion.isin([90, 0]))
            ].copy()
            force_ylim = 'best'

        get_main_result_plots(
            results[((results['framework'] == plot_params['framework']) | (results['framework'].isnull()))], 
            legend=False,
            nm_lines=plot_params['nm_lines'],
            draw_nm_region=plot_params['draw_nm_region'],
            any_or_corr=plot_params['any_or_corr'], 
            all_or_nem=plot_params['all_or_nem'],
            metrics=plot_params['metrics'], 
            sample_desc=plot_name,
            savefig=True, 
            force_ylim=force_ylim,
            figure_output_dir=figure_output_dir
        )

    elif plot_params['function'] == 'get_sample_size_plots':

        sample_size_results = all_results[
            (all_results.blocked_proportion.isin([90])) & 
            (all_results.algorithm.str.contains('dedupe'))
        ].copy()
        sample_size_results['framework'] = sample_size_results.framework + ' (' + sample_size_results.sample_size.astype(str) + ')'

        get_sample_size_plots(
            sample_size_results[(sample_size_results['framework'].str.contains(plot_params['framework']))], 
            any_or_corr=plot_params['any_or_corr'], 
            all_or_nem=plot_params['all_or_nem'],
            metrics=plot_params['metrics'],
            drop_record_linkage=plot_params['drop_record_linkage'],
            sample_desc=plot_name, 
            savefig=True,
            figure_output_dir=figure_output_dir
        )

    else:
        raise ValueError("plot function not defined")


def main(args):

    params = yaml.safe_load(open(args.config_file,'r'))
    all_results = pd.read_csv(args.results_file)

    nm_bs_results = get_name_match_bootstrap_results(
        args.nm_results_dir, 
        params['n_namematch_bootstrap_iterations'], 
        n_admin_rows_list=params['n_admin_rows'])

    all_results = pd.merge(
        all_results, 
        nm_bs_results, 
        on=['algorithm', 'ss_train', 'ss_eval', 'em_train', 'em_eval', 'iter'], 
        how='left'
    )

    all_results = all_results[
        (all_results.em_eval == 50) & (all_results.em_train == 50) & 
        (all_results.ss_train == all_results.ss_eval) & 
        (all_results.ss_train.isin(params['n_admin_rows']))
    ]

    all_results['N Admin Rows'] = all_results.ss_train.copy()
    all_results.loc[all_results.algorithm.str.contains('dedupe'), 'framework'] = "Deduplication"
    all_results.loc[all_results.algorithm.str.contains('record_linkage'), 'framework'] = 'Record Linkage'

    for plot_name, plot_params in params['plots_to_genearte'].items():
        create_figure(all_results, plot_name, plot_params, figure_output_dir=args.output_dir)

    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file')
    parser.add_argument('--nm_results_dir')
    parser.add_argument('--config_file')
    parser.add_argument('--output_dir')
    args = parser.parse_args()  

    logging.basicConfig(
        filename=f'{args.output_dir}/task.log', 
        filemode='w', 
        level=logging.INFO,
        format='%(levelname)s - %(message)s') 

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    main(args)
