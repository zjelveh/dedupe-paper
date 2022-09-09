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
from cluster_utils import cluster_edges, convert_predictions_to_all_names_with_clusterid

from automated_labeling_utils import console_label_with_budget


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


def main(sst, emt, budget, the_type, sample_size, iteration, dataset_dir, output_file, output_dir, failed_runs_file):
    
    em_eval = [emt]
    ss_eval = [sst]

    budget = int(budget)
    sample_size = int(sample_size)

    if (the_type == 'record_linkage') and (sample_size not in [15000, 150000]):
        return

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

    all_results = []
    all_bootstrapped_results = []
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
                admin_eval_path = os.path.join(dataset_dir, 'admin_evaluation__' + str(sse) + '__' + str(eme) + '.csv')
                admin_eval = readData(admin_eval_path)
                admin_eval_df = pd.read_csv(admin_eval_path)

                exp_eval_path = os.path.join(dataset_dir, 'experiment_evaluation__'  + str(sse) + '__' + str(eme) + '.csv')
                exp_eval = readData(exp_eval_path)
                exp_eval_df = pd.read_csv(exp_eval_path)
            except: 
                continue

            print('the_type:', the_type, 
                'sst:', sst, 
                'emt:', emt, 
                'budget:', budget,
                'iter:', iteration, 
                'sample_size:', sample_size,
                'eme:', eme, 
                'sse:', sse)
                            
            print('start preparing')
            try:
                
                if learn_model:

                    start_prepare_train = time.time()

                    if the_type=='record_linkage':
                        deduper = dedupe.RecordLink(variable_definition=var_defs, num_cores=8)
                        deduper.prepare_training(admin_train, exp_train, sample_size=sample_size)
                    else:
                        deduper = dedupe.Dedupe(variable_definition=var_defs, num_cores=8)
                        deduper.prepare_training(admin_train, sample_size=sample_size)

                    end_prepare_train = time.time()

                    # hopefully this doesn't throw off our runtime estimation too much
                    n_cp_1s = np.sum([1 for pair in deduper.active_learner.candidates if pair[0]['sid'] == pair[1]['sid']])
                    n_cp_1s__nem = np.sum([1 for pair in deduper.active_learner.candidates if pair[0]['sid'] == pair[1]['sid'] and 
                                            (pair[0]['first_name'] != pair[1]['first_name'] or 
                                            pair[0]['last_name'] != pair[1]['last_name'] or 
                                            pair[0]['date_of_birth'] != pair[1]['date_of_birth'])]) 

                    realized_blocked_proportion = deduper.active_learner.realized_blocked_proportion

                    label_info_path = output_dir + '/label_info__sst_' + str(sst) + '__emt_' + str(emt) + '__budget_' + str(budget) + \
                            '__sample_size_' + str(sample_size) + '__type_' + the_type + '__iter_' + str(iteration) + '.csv'
                
                    start_label = time.time()
                    # function we wrote to skip manual labeling and automatically supply the ground truth label
                    label_info = console_label_with_budget(deduper, budget=budget) 
                    end_label = time.time()
                
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

                filepath_details = f"sst_{sst}__emt_{emt}__budget_{budget}__sample_size_{sample_size}__type_{the_type}__iter_{iteration}__sse_{sse}__eme_{eme}.csv"
                
                if the_type=='deduplication':
                    filepath = f"{output_dir}/predicted_clusters__{filepath_details}"
                    cluster_membership = {}
                    for cluster_id, (records, scores) in enumerate(clustered_dupes):
                        for record_id, score in zip(records, scores):
                            cluster_membership[record_id] = {
                                "rowid": record_id,
                                "cluster_id": cluster_id,
                                "confidence_score": score
                            }

                    predicted_clusters = pd.DataFrame.from_dict(cluster_membership, orient='index')
                    end_predict = time.time()
                    predicted_clusters.to_csv(filepath, index=None)

                    an_w_cluster_id = convert_predictions_to_all_names_with_clusterid(exp_eval_df, admin_eval_df, predicted_clusters=predicted_clusters)

                else:
                    filepath = f"{output_dir}/predicted_matches__{filepath_details}"
                    predicted_matches = pd.DataFrame.from_records(clustered_dupes)
                    predicted_matches.columns = ['pair', 'confidence_score']
                    predicted_matches['rowid1'] = predicted_matches.pair.apply(lambda x: x[0])
                    predicted_matches['rowid2'] = predicted_matches.pair.apply(lambda x: x[1])
                    predicted_matches = predicted_matches.drop(columns={'pair'})

                    end_predict = time.time()
                    predicted_matches.to_csv(filepath, index=None)
                
                    an_w_cluster_id = convert_predictions_to_all_names_with_clusterid(exp_eval_df, admin_eval_df, predicted_matches=predicted_matches)

                print("getting training pair stats")
                n_distinct = len(deduper.training_pairs['distinct'])
                n_match = len(deduper.training_pairs['match'])

                print("evaluating")
                results_df_row = eval_utils.eval_predictions(
                    an_w_cluster_id, ss_train=sst, em_train=emt, ss_eval=sse, em_eval=eme, framework=the_type, 
                    budget=budget, dedupe_sample_size=sample_size, model_iter=iteration)

                results_df_row['cp_1s'] = n_cp_1s
                results_df_row['cp_1s__nem'] = n_cp_1s__nem
                results_df_row['labeled_0s'] = n_distinct
                results_df_row['labeled_1s'] = n_match
                results_df_row['labeled_base_rate'] = n_match / (n_match + n_distinct)
                results_df_row['realized_blocked_proportion'] = realized_blocked_proportion
                results_df_row['runtime_min__prepare_train'] = (end_prepare_train - start_prepare_train)/60
                results_df_row['runtime_min__label'] = (end_label - start_label)/60
                results_df_row['runtime_min__train'] = (end_train - start_train)/60
                results_df_row['runtime_min__predict'] = (end_predict - start_predict)/60
                all_results.append(results_df_row)
                
                print("bootstrap evaluating")
                bs_result_df = eval_utils.bootstrap_eval_predictions(
                    an_w_cluster_id, ss_train=sst, em_train=emt, ss_eval=sse, em_eval=eme, framework=the_type, 
                    budget=budget, dedupe_sample_size=sample_size, model_iter=iteration)
                all_bootstrapped_results.append(bs_result_df.copy())

            except:

                print("Unexpected error:", sys.exc_info()[0])
                
                new_failed_run = pd.DataFrame([[
                    the_type, sst, emt, budget, iteration, eme, sse, sample_size]], 
                    columns=['the_type', 'sst', 'emt', 'budget', 'iter', 'eme', 'sse', 'sample_size'])
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

            all_bootstraped_results_concat = pd.concat(all_bootstrapped_results, axis=0)
            all_bootstraped_results_concat.to_csv(output_file.replace('result', 'bootstrapped_result'), index=None)

    return 
                    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ss_train')
    parser.add_argument('--em_train')
    parser.add_argument('--budget')
    parser.add_argument('--type')
    parser.add_argument('--sample_size')
    parser.add_argument('--iteration')
    parser.add_argument('--dataset_dir')
    parser.add_argument('--output_file')
    parser.add_argument('--output_dir')
    parser.add_argument('--failed_runs_file')
    args = parser.parse_args()

    main(args.ss_train,
         args.em_train,
         args.budget,
         args.type,
         args.sample_size,
         args.iteration,
         args.dataset_dir,
         args.output_file,
         args.output_dir,
         args.failed_runs_file)

