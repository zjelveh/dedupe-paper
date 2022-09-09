
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


def add_confidence_intervals(all_results, all_bs_results):

    spec_cols = ['task', 'algorithm', 'ss_train', 'ss_eval', 'em_train', 'em_eval', 'model_iter', 
                 'framework', 'dedupe_sample_size', 'budget']
    metric_cols = ['tpr_any', 'prc_any', 'tnr_any', 'error_any', 'f1_any']

    all_results__confidence_intervals = (
        all_bs_results.groupby(spec_cols)[metric_cols]
        .quantile([.025, .975])
        .unstack().rename_axis(['metric', 'pctl'], axis=1).stack(level=0))

    all_results__confidence_intervals.columns = ['lower_bound', 'upper_bound']
    all_results__confidence_intervals = all_results__confidence_intervals
    all_results__confidence_intervals = all_results__confidence_intervals.unstack()
    all_results__confidence_intervals.columns = all_results__confidence_intervals.columns.swaplevel()
    all_results__confidence_intervals.columns = ['__'.join(col).strip() for col in all_results__confidence_intervals.columns.values]
    all_results__confidence_intervals = all_results__confidence_intervals.reset_index()

    all_results = pd.merge(all_results, all_results__confidence_intervals, on=spec_cols, how='left')
    all_results = all_results.set_index(spec_cols).sort_index(axis=1).reset_index()

    return all_results


def create_figure(all_results, plot_name, plot_params, figure_output_dir):

    logging.info(f"Creating the {plot_name} plot...")

    if plot_params['function'] == 'get_main_result_plots':

        if plot_params['dedupe_defaults']:
            results = all_results[
                (all_results.dedupe_sample_size == 0) |
                ((all_results.algorithm == 'dedupe (deduplication)') & (all_results.dedupe_sample_size == 1500)) |
                ((all_results.algorithm == 'dedupe (record_linkage)') & (all_results.dedupe_sample_size == 15000))
            ].copy()
            force_ylim = 'default'
        else:
            results = all_results[
                (all_results.dedupe_sample_size.isin([150000, 0]))
            ].copy()
            force_ylim = 'best'

        get_main_result_plots(
            results[((results['framework'] == plot_params['framework']) | (results['framework'] == ''))], 
            legend=False,
            benchmark_lines=plot_params['benchmark_lines'],
            benchmark_uncertainty_shading=plot_params['benchmark_uncertainty_shading'],
            any_or_corr=plot_params['any_or_corr'], 
            metrics=plot_params['metrics'], 
            sample_desc=plot_params['sample_desc'],
            savefig=True, 
            force_ylim=force_ylim,
            figure_output_dir=figure_output_dir
        )

    elif plot_params['function'] == 'get_sample_size_plots':

        sample_size_results = all_results[
            (all_results.algorithm.str.contains('dedupe'))
        ].copy()
        sample_size_results['framework'] = sample_size_results.framework + ' (' + sample_size_results.dedupe_sample_size.astype(str) + ')'

        get_sample_size_plots(
            sample_size_results[(sample_size_results['framework'].str.contains(plot_params['framework']))], 
            any_or_corr=plot_params['any_or_corr'], 
            metrics=plot_params['metrics'],
            sample_desc=plot_params['sample_desc'], 
            savefig=True,
            figure_output_dir=figure_output_dir
        )

    else:
        raise ValueError("plot function not defined")


def get_median_indicator(results_df, median_metric):

    gb_cols = [
        'task', 'algorithm', 'framework', 'ss_train', 'ss_eval', 
        'em_train', 'em_eval', 'budget', 'dedupe_sample_size'
    ]

    assert results_df.groupby(gb_cols).size().max() <= results_df.model_iter.nunique()

    results_df['median'] = results_df.groupby(gb_cols)[median_metric].transform('median')
    results_df['dist_from_median'] = (results_df[median_metric] - results_df['median']).abs()
    results_df['is_median'] = results_df.groupby(gb_cols)['dist_from_median'].rank(method='first')
    results_df['is_median'] = results_df.is_median == 1

    results_df = results_df.drop(columns=['median', 'dist_from_median'])

    return results_df


def main(args):

    params = yaml.safe_load(open(args.config_file,'r'))

    all_results = pd.read_csv(args.results_file)
    all_results = all_results.drop(columns=all_results.filter(regex='runtime').columns)
    all_results['task'] = 'ojin'
    all_results['N Admin Rows'] = all_results.ss_train.copy()
    all_results.loc[all_results.algorithm == 'namematch', 'framework'] = ''

    all_results = get_median_indicator(all_results, params['median_metric'])

    all_bs_results = pd.read_csv(args.bootstrap_results_file)
    all_bs_results['task'] = 'ojin'
    all_bs_results.loc[all_bs_results.algorithm == 'namematch', 'framework'] = ''

    all_results = add_confidence_intervals(all_results, all_bs_results)

    for plot_name, plot_params in params['plots_to_genearte'].items():
        create_figure(all_results, plot_name, plot_params, figure_output_dir=args.output_dir)

    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file')
    parser.add_argument('--bootstrap_results_file')
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
