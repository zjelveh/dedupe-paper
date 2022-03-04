import argparse
import pandas as pd
import dedupe
import json
import numpy as np
import os
import csv
import re
import time
import sys
import logging
from unidecode import unidecode

sys.path.append('../utils/')
import eval_utils
from cluster_utils import cluster_edges


def readData(filename):
    data_d = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = [(k, v) for (k, v) in row.items()]
            row_id = int(row['rowid'])
            data_d[row_id] = dict(clean_row)

    return data_d

def age_diff(f1, f2):
    return(abs(float(f1) - float(f2)))


def main(dataset_version, sst, emt, budget, the_type, sample_size, blocked_proportion, iteration, dataset_dir, output_file, output_dir, failed_runs_file):
    
    em_eval = [50]
    ss_eval = [5000, 50000, 200000]

    var_defs = [
        {'field': 'first_name', 'type': 'String'},
        {'field': 'last_name', 'type': 'String'},
        {'field': 'date_of_birth', 'type': 'DateTime', 'yearfirst':True},
        {'field': 'date_of_birth', 'type': 'String'},
        {'field': 'middle_initial', 'type': 'Exact'},
        {'field': 'race', 'type': 'Exact'},
        {'field': 'sex', 'type': 'Exact'},
        {'field': 'age_in_2020', 'type': 'Custom', 'comparator': age_diff}
    ]

    budget = int(budget)
    sample_size = int(sample_size)
    bp_float = float(blocked_proportion)/100

    all_results = []
    try:
        admin_train = readData(os.path.join(dataset_dir, 'admin_training__' + str(sst) + '__' + str(emt) + '.csv'))
        exp_train = readData(os.path.join(dataset_dir, 'experiment_training__' + str(sst) + '__' + str(emt) + '.csv'))
    except: 
        return 
    learn_model = True
    try:
        del deduper
    except:
        pass     
    for eme in em_eval:
        for sse in ss_eval:

            try:
                admin_eval = readData(os.path.join(dataset_dir, 'admin_evaluation__' + str(sse) + '__' + str(eme) + '.csv'))
                exp_eval = readData(os.path.join(dataset_dir, 'experiment_evaluation__'  + str(sse) + '__' + str(eme) + '.csv'))
            except: 
                continue

            print('the_type:', the_type, 
                'dataset_version:', dataset_version,
                'sst:', sst, 
                'emt:', emt, 
                'budget:', budget,
                'iter:', iteration, 
                'sample_size:', sample_size,
                'blocked_proportion:', blocked_proportion,
                'eme:', eme, 
                'sse:', sse)
                            
            print('start preparing')
            try:
                
                if learn_model:

                    start_prepare_train = time.time()

                    if the_type=='record_linkage':
                        deduper = dedupe.RecordLink(variable_definition=var_defs, num_cores=8)
                        deduper.prepare_training(admin_train, exp_train, sample_size=sample_size, blocked_proportion=bp_float)
                    else:
                        deduper = dedupe.Dedupe(variable_definition=var_defs, num_cores=8)
                        deduper.prepare_training(admin_train, sample_size=sample_size, blocked_proportion=bp_float)

                    end_prepare_train = time.time()

                    # hopefully this doesn't throw off our runtime estimation too much
                    n_cp_1s = np.sum([1 for pair in deduper.active_learner.candidates if pair[0]['sid'] == pair[1]['sid']])
                    n_cp_1s__nem = np.sum([1 for pair in deduper.active_learner.candidates if pair[0]['sid'] == pair[1]['sid'] and 
                                            (pair[0]['first_name'] != pair[1]['first_name'] or 
                                            pair[0]['last_name'] != pair[1]['last_name'] or 
                                            pair[0]['date_of_birth'] != pair[1]['date_of_birth'])]) 

                    realized_blocked_proportion = deduper.active_learner.realized_blocked_proportion

                    label_info_path = output_dir + '/label_info__' + dataset_version + '__sst_' + str(sst) + '__emt_' + str(emt) + '__budget_' + str(budget) + \
                            '__sample_size_' + str(sample_size) + '__block_prop_' + blocked_proportion + '__type_' + the_type + '__iter_' + str(iteration) + '.csv'
                
                    start_label = time.time()
                    # function we wrote to skip manual labeling and automatically supply the ground truth label
                    label_info = dedupe.console_label_with_budget(deduper, budget=budget) 
                    end_label = time.time()
                    print('finished labeling')
                
                    start_train = time.time()
                    deduper.train(recall=.99)
                    end_train = time.time()
                    learn_model = False

                    label_info.to_csv(label_info_path, index=None)

                start_predict = time.time()

                if the_type=='record_linkage':
                    clustered_dupes = deduper.join(admin_eval, exp_eval, 0.5, 'many-to-one')
                else:
                    for k, v in exp_eval.items():
                        admin_eval[k] = v

                    print('partitioning')

                    clustered_dupes = deduper.partition(admin_eval, 0.5)

                print("clustered_dupes exists")

                links_filepath = output_dir + '/links__' + dataset_version + '__sst_' + str(sst) + '__emt_' + str(emt) + '__budget_' + str(budget) + \
                        '__sample_size_' + str(sample_size) + '__block_prop_' + blocked_proportion + '__type_' + the_type + '__iter_' + str(iteration) + '__sse_' + str(sse) + '__eme_' + str(eme) + '.csv'
                
                if the_type=='dedupe':
                    cluster_membership = {}
                    for cluster_id, (records, scores) in enumerate(clustered_dupes):
                        for record_id, score in zip(records, scores):
                            cluster_membership[record_id] = {
                                "rowid": record_id,
                                "cluster_id": cluster_id,
                                "confidence_score": score
                            }

                    predictions = pd.DataFrame.from_dict(cluster_membership, orient='index')
                    end_predict = time.time()
                    predictions.to_csv(links_filepath, index=None)

                else:

                    links = pd.DataFrame.from_records(clustered_dupes)
                    links.columns = ['pair', 'confidence_score']
                    links['rowid1'] = links.pair.apply(lambda x: x[0])
                    links['rowid2'] = links.pair.apply(lambda x: x[1])
                    links = links.drop(columns={'pair'})

                    cluster_membership = cluster_edges(clustered_dupes)

                    predictions = pd.DataFrame.from_dict(cluster_membership, orient='index')
                    predictions = predictions.reset_index()
                    predictions.columns = ['rowid', 'cluster_id']
                    end_predict = time.time()
                    links.to_csv(links_filepath, index=None)

                print("getting training pairs")
                
                n_distinct = len(deduper.training_pairs['distinct'])
                n_match = len(deduper.training_pairs['match'])

                print("writing output")

                ae = pd.read_csv(os.path.join(dataset_dir, 'admin_evaluation__' + str(sse) + '__' + str(eme) + '.csv'))
                exp = pd.read_csv(os.path.join(dataset_dir, 'experiment_evaluation__' + str(sse) + '__' + str(eme) + '.csv'))
                both = pd.concat([ae, exp], axis=0)
                both = both.merge(predictions, on='rowid', how='left')
                both.loc[(both.dataset == 'admin_evaluation') & (both.cluster_id.isna()), 'cluster_id'] = -2
                both.loc[(both.dataset == 'experiment_evaluation') & (both.cluster_id.isna()), 'cluster_id'] = -1
                both['cluster_id'] = both.cluster_id.astype(int)

                match_file = output_dir + '/cw__' + dataset_version + '__sst_' + str(sst) + '__emt_' + str(emt) + '__budget_' + str(budget) + \
                        '__sample_size_' + str(sample_size) + '__block_prop_' + blocked_proportion + '__type_' + the_type + '__iter_' + str(iteration) + '__sse_' + str(sse) + '__eme_' + str(eme) + '.csv'
                both_to_write = both[['rowid', 'cluster_id', 'sid', 'dataset']].copy()
                both_to_write.to_csv(match_file, index=None)
                
                evals = eval_utils.eval_predictions(both)
                evals['dataset_version'] = dataset_version
                evals['em_train'] = emt
                evals['em_eval'] = eme
                evals['ss_train'] = sst
                evals['ss_eval'] = sse
                evals['type'] = the_type
                evals['budget'] = budget
                evals['iter'] = iteration
                evals['sample_size'] = sample_size
                evals['blocked_proportion'] = blocked_proportion
                evals['cp_1s'] = n_cp_1s
                evals['cp_1s__nem'] = n_cp_1s__nem
                evals['labeled_0s'] = n_distinct
                evals['labeled_1s'] = n_match
                evals['labeled_base_rate'] = n_match / (n_match + n_distinct)
                evals['realized_blocked_proportion'] = realized_blocked_proportion
                evals['runtime_min__prepare_train'] = (end_prepare_train - start_prepare_train)/60
                evals['runtime_min__label'] = (end_label - start_label)/60
                evals['runtime_min__train'] = (end_train - start_train)/60
                evals['runtime_min__predict'] = (end_predict - start_predict)/60
                all_results.append(evals)

            except:

                print("Unexpected error:", sys.exc_info()[0])
                
                new_failed_run = pd.DataFrame([[
                    the_type, dataset_version, sst, emt, budget, iteration, eme, sse, sample_size, blocked_proportion]], 
                    columns=['the_type', 'dataset_version', 'sst', 'emt', 'budget', 'iter', 'eme', 'sse', 'sample_size', 'blocked_proportion'])
                if os.path.exists(failed_runs_file):
                    failed_runs = pd.read_csv(failed_runs_file)
                    failed_runs = pd.concat([failed_runs, new_failed_run], axis=0)
                else:
                    failed_runs = new_failed_run
                failed_runs.to_csv(failed_runs_file, index=None)
                
        if len(all_results) > 0:
            all_results_concat = pd.concat(all_results, axis=0)
            all_results_concat.to_csv(output_file, index=None)
            print(all_results_concat)
        
                    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_version')
    parser.add_argument('--ss_train')
    parser.add_argument('--em_train')
    parser.add_argument('--budget')
    parser.add_argument('--type')
    parser.add_argument('--sample_size')
    parser.add_argument('--iteration')
    parser.add_argument('--blocked_proportion')
    parser.add_argument('--dataset_dir')
    parser.add_argument('--output_file')
    parser.add_argument('--output_dir')
    parser.add_argument('--failed_runs_file')
    args = parser.parse_args()

    main(args.dataset_version,
         args.ss_train,
         args.em_train,
         args.budget,
         args.type,
         args.sample_size,
         args.blocked_proportion,
         args.iteration,
         args.dataset_dir,
         args.output_file,
         args.output_dir,
         args.failed_runs_file)

