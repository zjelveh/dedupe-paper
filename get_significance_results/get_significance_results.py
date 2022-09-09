
import pandas as pd
import os
import yaml
import numpy as np
import argparse
import logging
import sys

from pandas.api.types import CategoricalDtype


def filter_and_clean_results_df(results_df):

    results_df = results_df[
        (results_df.dedupe_sample_size == 0) |
        ((results_df.algorithm == 'dedupe (deduplication)') & (results_df.dedupe_sample_size == 1500)) |
        ((results_df.algorithm == 'dedupe (record_linkage)') & (results_df.dedupe_sample_size == 15000)) |
        (results_df.dedupe_sample_size == 150000)
    ].copy()

    results_df['intervention'] = results_df.algorithm.str.extract('(^[a-z]+) ?')
    results_df.loc[results_df.dedupe_sample_size == 150000, 'intervention'] = results_df.intervention + " (optimized)"
    results_df.loc[(results_df.dedupe_sample_size != 150000) & 
                     (results_df.dedupe_sample_size != 0), 'intervention'] = results_df.intervention + " (default)"

    return results_df


def process_point_estimate_results(all_results, significance_metric):

    point_est_df = filter_and_clean_results_df(all_results)

    gb_cols = ['intervention', 'ss_train', 'framework', 'budget']
    lower_is_better = 'error' in significance_metric

    point_est_df['median'] = point_est_df.groupby(gb_cols)[significance_metric].transform('median')
    point_est_df['dist_from_median'] = (point_est_df[significance_metric] - point_est_df['median']).abs()
    point_est_df['is_median'] = point_est_df.groupby(gb_cols)['dist_from_median'].rank(method='first')
    point_est_df['is_median'] = point_est_df.is_median == 1

    point_est_df['is_best'] = point_est_df.groupby(gb_cols)[significance_metric].rank(method='first', ascending=lower_is_better)
    point_est_df['is_best'] = point_est_df.is_best == 1

    point_est_df['is_worst'] = point_est_df.groupby(gb_cols)[significance_metric].rank(method='first', ascending=(not lower_is_better))
    point_est_df['is_worst'] = point_est_df.is_worst == 1

    point_est_df['is_worst_case_scenario'] = (((point_est_df.is_worst == 1) & (point_est_df.intervention == 'dedupe (optimized)')) | 
                                                ((point_est_df.is_best == 1) & (point_est_df.intervention != 'dedupe (optimized)')))

    point_est_df = point_est_df.drop(columns=['median', 'dist_from_median', 'is_best', 'is_worst'])

    return point_est_df


def process_bootstrap_results(all_bs_results, significance_metric, point_estimate_df):

    sig_test_df = filter_and_clean_results_df(all_bs_results)

    treatment_definition_dict = {
        'namematch': ['ss_train'],
        'fastlink': ['ss_train', 'framework'],
        'dedupe': ['ss_train', 'framework', 'budget']
    }

    merge_cols = ['intervention'] + treatment_definition_dict['dedupe'] + ['model_iter']
    len_before = len(sig_test_df)
    sig_test_df = pd.merge(sig_test_df, point_estimate_df[merge_cols + ['is_median', 'is_worst_case_scenario']], on=merge_cols)
    len_after = len(sig_test_df)
    assert len_before == len_after

    addtl_cols = [significance_metric, 'model_iter', 'bs_iter', 'is_median', 'is_worst_case_scenario']
    optimized_dedupe_sig_test_df = sig_test_df[sig_test_df.intervention == 'dedupe (optimized)'][treatment_definition_dict['dedupe'] + addtl_cols].copy()

    comparison_algorithm_list = ['default_dedupe', 'fastlink', 'namematch']

    all_comparison_sig_test_dfs = []
    for algorithm in comparison_algorithm_list:

        intervention = algorithm if algorithm != 'default_dedupe' else 'dedupe (default)'
        alg_lookup = algorithm if algorithm != 'default_dedupe' else 'dedupe'
        
        comparison_sig_test_df = sig_test_df[sig_test_df.intervention == intervention][treatment_definition_dict[alg_lookup] + addtl_cols].copy()

        comparison_sig_test_df = pd.merge(
            optimized_dedupe_sig_test_df,
            comparison_sig_test_df,
            on=[c for c in treatment_definition_dict['dedupe'] if c in comparison_sig_test_df.columns] + ['bs_iter'],
            how='left',
            suffixes=['__optimized_dedupe', f'__{algorithm}']
        )
        comparison_sig_test_df.columns = [c if algorithm not in c else c.replace(algorithm, 'comparison_alg') for c in comparison_sig_test_df.columns]
        
        if algorithm != 'namematch':
            comparison_sig_test_df['performance_difference'] = \
                comparison_sig_test_df.error_any__optimized_dedupe - comparison_sig_test_df.error_any__comparison_alg
        else:
            comparison_sig_test_df['performance_difference'] = \
                comparison_sig_test_df.error_any__comparison_alg - comparison_sig_test_df.error_any__optimized_dedupe

        all_comparison_sig_test_dfs.append(comparison_sig_test_df.copy())

    sig_test_df = pd.concat(all_comparison_sig_test_dfs, keys=comparison_algorithm_list)

    sig_test_df = sig_test_df.reset_index(level=0).rename(columns={'level_0':'comparison_alg'}).reset_index(drop=True)

    sig_test_df['median_comparison'] = (sig_test_df.is_median__optimized_dedupe == 1) & (sig_test_df.is_median__comparison_alg == 1)
    sig_test_df['worst_case_scenario_comparison'] = \
        (sig_test_df.is_worst_case_scenario__optimized_dedupe == 1) & (sig_test_df.is_worst_case_scenario__comparison_alg == 1)
    sig_test_df['crossproduct_comparison'] = True

    sig_test_df = sig_test_df[[
        'framework', 'ss_train', 'budget', 'comparison_alg', 'model_iter__optimized_dedupe', 'model_iter__comparison_alg', 
        'bs_iter', f"{significance_metric}__optimized_dedupe", f'{significance_metric}__comparison_alg', 'performance_difference', 
        'median_comparison', 'worst_case_scenario_comparison', 'crossproduct_comparison']]
    sig_test_df = sig_test_df.sort_values(['framework', 'ss_train', 'budget', 'comparison_alg', 
                                           'model_iter__optimized_dedupe', 'model_iter__comparison_alg', 'bs_iter'])

    if 'error' in significance_metric:
        sig_test_df['performance_difference'] = -1 * sig_test_df['performance_difference']

    return sig_test_df


def get_significance_test_results(sig_test_df, significance_level):

    gb_keys = ['framework', 'ss_train', 'budget', 'comparison_alg', 'model_iter__optimized_dedupe', 'model_iter__comparison_alg']

    sig_test_results_df = sig_test_df.groupby(gb_keys).agg({
        'performance_difference': lambda x: x.quantile(significance_level),
        'median_comparison': max,
        'worst_case_scenario_comparison': max,
        'crossproduct_comparison': max
    }).rename(columns={'performance_difference': 'performance_difference_at_pctl'}).reset_index()

    sig_test_results_df['significantly_better'] = (sig_test_results_df.performance_difference_at_pctl > 0)
    
    sig_test_results_df__crossproduct_comparison_share_significant = \
        sig_test_results_df.groupby(['framework', 'ss_train', 'budget', 'comparison_alg']).significantly_better.mean().reset_index()
    
    sig_test_results_df__median_comparison = \
        sig_test_results_df[sig_test_results_df.median_comparison == 1].copy()
    
    sig_test_results_df__worst_case_scenario_comparison = \
        sig_test_results_df[sig_test_results_df.worst_case_scenario_comparison == 1].copy()

    sig_test_results_df = format_significance_test_results(
        sig_test_results_df__crossproduct_comparison_share_significant,
        sig_test_results_df__median_comparison,
        sig_test_results_df__worst_case_scenario_comparison
    )
    return sig_test_results_df


def format_significance_test_results(cross_product_comparison, median_comparison, worst_case_comparison):

    all_ct = []
    for comparison in ['Worst-case comparison', 'Median comparison', 'Share of all possible comparisons']:
        
        for framework in ['deduplication', 'record_linkage']:

            comparison_dict = {
                'Median comparison' : median_comparison, 
                'Worst-case comparison' : worst_case_comparison, 
                'Share of all possible comparisons' : cross_product_comparison
            }
            comparison_df = comparison_dict[comparison].copy()
            comparison_cat_order = CategoricalDtype(categories=list(comparison_dict.keys()), ordered=True)

            comparison_df = comparison_df[comparison_df.framework == framework].copy()

            question_map = {
                'default_dedupe': "Is optimized dedupe better than default dedupe?",
                'fastlink': "Is optimized dedupe better than fastLink?",
                'namematch': "Is Name Match better than optimized dedupe?"
            }
            comparison_df['question'] = comparison_df.comparison_alg.map(question_map)
            row_order = list(question_map.values())
            question_cat_order = CategoricalDtype(categories=list(question_map.values()), ordered=True)
            
            comparison_df = comparison_df.rename(columns={'ss_train':'admin_dataset_size'})

            ct = pd.crosstab(
                [comparison_df.question],
                [comparison_df.admin_dataset_size, comparison_df.budget],
                values=comparison_df.significantly_better,
                aggfunc='mean'
            )
            ct = ct.loc[row_order]

            if ((ct < 1) & (ct > 0)).any().any():
                ct = (ct * 100).round(0).astype(int).astype(str) + '%'
            else:
                ct = ct.astype(int).replace({1:'*', 0:''})
                
            ct['comparison'] = comparison
            ct['framework'] = framework
            ct = ct.reset_index()
            ct['comparison'] = ct.comparison.astype(comparison_cat_order)
            ct['question'] = ct.question.astype(question_cat_order)
            ct = ct.set_index(['framework', 'question', 'comparison'])
            all_ct.append(ct.copy())
        
    sig_test_results_df = pd.concat(all_ct).sort_index()

    return sig_test_results_df


def main(args):

    params = yaml.safe_load(open(args.config_file,'r'))

    all_results = pd.read_csv(args.results_file)
    all_results = all_results.drop(columns=all_results.filter(regex='runtime').columns)
    all_results['task'] = 'ojin'
    all_results['N Admin Rows'] = all_results.ss_train.copy()
    all_results.loc[all_results.algorithm == 'namematch', 'framework'] = ''

    all_bs_results = pd.read_csv(args.bootstrap_results_file)
    all_bs_results['task'] = 'ojin'
    all_bs_results.loc[all_bs_results.algorithm == 'namematch', 'framework'] = ''

    point_estimate_df = process_point_estimate_results(all_results, params['significance_metric'])
    sig_test_df = process_bootstrap_results(all_bs_results, params['significance_metric'], point_estimate_df)

    sig_test_results_df = get_significance_test_results(sig_test_df, params['significance_level'])

    with pd.ExcelWriter(args.output_file) as excel_writer:
        sig_test_results_df.loc['deduplication'].to_excel(excel_writer, sheet_name='Deduplication')
        sig_test_results_df.loc['record_linkage'].to_excel(excel_writer, sheet_name='Record Linkage')

    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file')
    parser.add_argument('--bootstrap_results_file')
    parser.add_argument('--config_file')
    parser.add_argument('--output_file')
    args = parser.parse_args()  

    logging.basicConfig(
        filename=f'{os.path.dirname(args.output_file)}/task.log', 
        filemode='w', 
        level=logging.INFO,
        format='%(levelname)s - %(message)s') 
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    main(args)
