import pandas as pd
import numpy as np
import argparse
import yaml
import os

import sys
sys.path.append('/projects/2017-007-namematch/plosOne/melissa_work/repo/utils/')
from eval_utils import eval_predictions
from cluster_utils import cluster_edges

def concat_existing_result_files(nm_dir, dedupe_dir, params):

    dedupe_results_files = [os.path.join(dedupe_dir, f) for f in os.listdir(dedupe_dir) if 'result' in f]
    nm_results_files = [os.path.join(nm_dir, 'results.csv')]
    
    dedupe_results = pd.concat([pd.read_csv(r) for r in dedupe_results_files])
    dedupe_results['algorithm'] = 'dedupe'
    dedupe_results['budget'] = dedupe_results['budget'].astype(int)
    dedupe_results['sample_size'] = dedupe_results['sample_size'].astype(int)
    dedupe_results['blocked_proportion'] = dedupe_results['blocked_proportion'].astype(int)
    dedupe_results['algorithm'] = dedupe_results['algorithm'].astype(str) + ' (' + dedupe_results.type + ')'
    dedupe_results['runtime_min'] = dedupe_results.filter(regex='runtime_').sum(axis=1)

    nm_results = pd.concat([pd.read_csv(r) for r in nm_results_files])
    nm_results['algorithm'] = 'namematch'
    nm_results['budget'] = params['default_value']
    nm_results['sample_size'] = params['default_value']
    nm_results['blocked_proportion'] = params['default_value']
    nm_results['dataset_version'] = params['dataset_version']
    nm_results['runtime_min'] = nm_results.filter(regex='runtime_').sum(axis=1)
    
    results = pd.concat([dedupe_results, nm_results], sort=False, join='outer')
    results['algorithm'] = results.algorithm.str.replace('(dedupe)', '(deduplication)', regex=False)

    results['f1_corr'] = (2 * (results.prc_corr * results.tpr_corr)) / (results.prc_corr + results.tpr_corr)
    results['f1_any'] = (2 * (results.prc_any * results.tpr_any)) / (results.prc_any + results.tpr_any)
    
    return results


def get_all_additional_results(dataset_dir, nm_dir, dedupe_dir, fl_dir, params):

    nem_results_list = []
    results_by_race_list = []
    for alg in ['namematch', 'dedupe']:
    
        blocked_proportion_grid = [None]
        sample_size_grid = [None]
        budget_grid = [None]
        if alg == 'dedupe':
            blocked_proportion_grid = params['blocked_proportion_grid']
            sample_size_grid = params['sample_size_grid']
            budget_grid = params['budget_grid']

        for sse in params['ss_grid']:
            for eme in params['em_grid']:
                exp_eval = pd.read_csv(os.path.join(dataset_dir, f'experiment_evaluation__{sse}__{eme}.csv'))
                admin_eval = pd.read_csv(os.path.join(dataset_dir, f'admin_evaluation__{sse}__{eme}.csv'))
                sst_grid = params['ss_grid']
                emt_grid = params['em_grid']
                for sst in sst_grid:
                    for emt in emt_grid:
                        for the_type in params['type_grid']:
                            if alg == 'namematch' and the_type == 'record_linkage':
                                continue
                            alg_str = alg
                            if alg != 'namematch':
                                alg_str = f"{alg} ({the_type.replace('dedupe', 'deduplication')})"
                            for i in params['iteration_grid']:
                                for blocked_proportion in blocked_proportion_grid:
                                    for sample_size in sample_size_grid: 
                                        for budget in budget_grid:

                                            try:
                                                nem_metrics, metrics_by_race = get_additional_results(
                                                    exp_eval, admin_eval, nm_dir, dedupe_dir, params, alg, 
                                                    sst, emt, sse, eme, the_type, i, blocked_proportion, sample_size, budget)

                                                metrics_by_race = pd.concat(metrics_by_race).reset_index(level=1, drop=True)
                                                metrics_by_race.index.name = 'race'
                                                metrics_by_race = metrics_by_race.reset_index()
                                                metrics_by_race['algorithm'] = alg_str
                                                metrics_by_race['dataset_version'] = params['dataset_version']
                                                metrics_by_race['ss_train'] = sst
                                                metrics_by_race['em_train'] = emt
                                                metrics_by_race['ss_eval'] = sse
                                                metrics_by_race['em_eval'] = eme
                                                metrics_by_race['type'] = the_type
                                                metrics_by_race['iter'] = i
                                                metrics_by_race['blocked_proportion'] = blocked_proportion if blocked_proportion is not None else params['default_value']
                                                metrics_by_race['sample_size'] = sample_size if sample_size is not None else params['default_value']
                                                metrics_by_race['budget'] = budget if budget is not None else params['default_value']

                                                addtl_results = {
                                                    'algorithm': alg_str,
                                                    'dataset_version': params['dataset_version'],
                                                    'ss_train': sst,
                                                    'em_train': emt,
                                                    'ss_eval': sse,
                                                    'em_eval': eme,
                                                    'type': the_type,
                                                    'iter': i,
                                                    'blocked_proportion': blocked_proportion if blocked_proportion is not None else params['default_value'], 
                                                    'sample_size': sample_size if sample_size is not None else params['default_value'], 
                                                    'budget': budget if budget is not None else params['default_value'],
                                                    'tpr_any__nem': nem_metrics.tpr_any.iloc[0],
                                                    'prc_any__nem': nem_metrics.prc_any.iloc[0],
                                                    'tnr_any__nem': nem_metrics.tnr_any.iloc[0],
                                                    'error_any__nem': nem_metrics.error_any.iloc[0],
                                                    'acc_any__nem': nem_metrics.acc_any.iloc[0],
                                                    'tpr_corr__nem': nem_metrics.tpr_corr.iloc[0],
                                                    'prc_corr__nem': nem_metrics.prc_corr.iloc[0],
                                                    'tpr_pair__nem': nem_metrics.tpr_pair.iloc[0],
                                                    'prc_pair__nem': nem_metrics.prc_pair.iloc[0],
                                                    'tnr_pair__nem': nem_metrics.tnr_pair.iloc[0],
                                                    'error_pair__nem': nem_metrics.error_pair.iloc[0]
                                                }

                                                nem_results_list.append(addtl_results.copy())
                                                results_by_race_list.append(metrics_by_race.copy())
                                            
                                            except:
                                                pass

    nem_results = pd.DataFrame.from_records(nem_results_list)
    results_by_race = pd.concat(results_by_race_list)

    nem_results['f1_corr__nem'] = (2 * (nem_results.prc_corr__nem * nem_results.tpr_corr__nem)) / (nem_results.prc_corr__nem + nem_results.tpr_corr__nem)
    nem_results['f1_any__nem'] = (2 * (nem_results.prc_any__nem * nem_results.tpr_any__nem)) / (nem_results.prc_any__nem + nem_results.tpr_any__nem)
    results_by_race['f1_corr'] = (2 * (results_by_race.prc_corr * results_by_race.tpr_corr)) / (results_by_race.prc_corr + results_by_race.tpr_corr)
    results_by_race['f1_any'] = (2 * (results_by_race.prc_any * results_by_race.tpr_any)) / (results_by_race.prc_any + results_by_race.tpr_any)

    return nem_results, results_by_race


def get_additional_results(exp_eval, admin_eval, nm_dir, dedupe_dir, params, alg, sst, emt, sse, eme, the_type, i, bp, sample_size, budget):

    # get all-names like table
    if alg == 'namematch':
        an = pd.read_csv(os.path.join(nm_dir, f'spec__{sst}_{emt}_{sse}_{eme}_{i}/output_temp/all_names_with_clusterid.csv'), low_memory=False)
        an['rowid'] = an.record_id.str.split('__', expand=True)[[1]]
        an = an.rename(columns={'dob':'date_of_birth'})
        an = an[an.dataset != 'ADMIN_TRAINING']
        an['dataset'] = an['dataset'].str.lower()

    elif alg == 'dedupe' and the_type == 'dedupe':
        an = pd.read_csv(os.path.join(dedupe_dir, f"cw__{params['dataset_version']}__sst_{sst}__emt_{emt}__budget_{budget}__sample_size_{sample_size}__block_prop_{bp}__type_dedupe__iter_{i}__sse_{sse}__eme_{eme}.csv"), low_memory=False)
        an = pd.concat([
            pd.merge(an, admin_eval[['rowid', 'first_name', 'last_name', 'date_of_birth', 'race']], on='rowid'),
            pd.merge(an, exp_eval[['rowid', 'first_name', 'last_name', 'date_of_birth', 'race']], on='rowid')
        ])
    else:
        raise
    
    an['namedob'] = an.first_name + ' ' + an.last_name + ' ' + an.date_of_birth
    an_nem = an.drop_duplicates(subset=['first_name', 'last_name', 'date_of_birth'])

    nem_results = eval_predictions(an_nem)

    results_by_race = {}
    for race in list(an[an.race.notnull()].race.unique()):
        race_metrics = eval_predictions(an[an.race == race])
        race_metrics['n_an_rows'] = len(an[an.race == race])
        race_metrics['pct_an_rows'] = (an.race == race).mean()
        results_by_race[race] = race_metrics.copy()

    return nem_results, results_by_race


def main(dataset_dir, nm_dir, dedupe_dir, params, output_file): 

    keys = ['dataset_version', 'em_train', 'em_eval', 'ss_train', 'ss_eval', 
            'budget', 'iter', 'sample_size', 'blocked_proportion', 'algorithm', 'type']

    existing_results = concat_existing_result_files(nm_dir, dedupe_dir, params)

    nem_results, results_by_race = get_all_additional_results(dataset_dir, nm_dir, dedupe_dir, params)
    print(nem_results.shape)

    results = pd.merge(
        existing_results, 
        nem_results,
        on=keys, 
        how='left'
    )
    print(results.shape)

    results = results.sort_values(['em_train', 'em_eval', 'ss_train', 'ss_eval', 'budget', 'iter'])

    results.to_csv(output_file, index=False)
    results_by_race.to_csv(output_file.replace('all_results', 'all_results_by_race'), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir_path')
    parser.add_argument('--nm_result_dir_path')
    parser.add_argument('--dedupe_result_dir_path')
    parser.add_argument('--config_file')
    parser.add_argument('--output_file')
    args = parser.parse_args()

    params = yaml.safe_load(open(args.config_file,'r'))

    dataset_dir = args.dataset_dir_path.replace('dataset_version', params['dataset_version'])

    main(dataset_dir,
         args.nm_result_dir_path,
         args.dedupe_result_dir_path,
         params,
         args.output_file)
