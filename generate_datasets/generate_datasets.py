import pandas as pd
import numpy as np
import argparse
import joblib
import yaml
import os
import sys
import logging
import random
from collections import defaultdict


def preprocess_data(df, col_with_lineage):
    
    df = df.copy()
    
    df['sid'] = 'SID' + df.sid.astype(int).astype(str)

    # extract lineage and clean middle name/initial
    df['middle_name'] = df.middle_name.str.extract('([A-Za-z ]+)')
    df['lineage'] = df[col_with_lineage].str.extract(' (JR|SR|II|III|IV|V)$')
    df['middle_name'] = df[col_with_lineage].str.replace(' (JR|SR|II|III|IV|V)$', '', regex=True).str.strip()
    df['middle_initial'] = df.middle_name.str.slice(0, 1)
    
    # get rid of very very low count U and D categories
    df.loc[df.sex.isin(['F', 'M']) == False, 'sex'] = np.NaN
    
    # add age
    df['age_in_2020'] = (pd.to_datetime('2020-01-01') - pd.to_datetime(df.date_of_birth)).astype('<m8[Y]').astype(int)
    
    # add a duplicate of sid that will be stripped for exp and eval datasets
    df['uid_sid'] = df.sid.copy()

    df = df[['rowid', 'first_name', 'last_name', 'date_of_birth', 'age_in_2020',
             'middle_name', 'middle_initial', 'lineage', 'race', 'sex', 'sid', 'uid_sid']]
    
    return df


def n_records_per_person(df):
    
    df = df.copy()
    return df.groupby('sid').size()


def share_singletons(df):
    
    df = df.copy()
    
    return (n_records_per_person(df) == 1).mean()


def add_stat_columns(df):

    df = df.copy()

    df['name_dob'] = df.first_name + ' ' + df.last_name + ' ' + df.date_of_birth
    df['sid_name_dob'] = df.sid + ' ' + df.name_dob
    df['n_records'] = df.groupby('sid').rowid.transform('nunique')
    df['n_uq_namedob'] = df.groupby('sid').name_dob.transform('nunique')
    df['n_records_per_sid_name_dob'] = df.groupby('sid_name_dob').rowid.transform('nunique')
    df['any_non_exact_match'] = (df.n_uq_namedob > 1).astype(int)
    df['all_non_exact_match'] = ((df.n_uq_namedob > 1) & (df.n_uq_namedob == df.n_records)).astype(int)

    return df


def train_eval_split(sid_type_df_dict, seed):
    
    train_sid_type_df_dict = {}
    eval_sid_type_df_dict = {}
    
    for sid_type, sid_type_df in sid_type_df_dict.items():
        
        train_sids = sid_type_df[['sid', 'n_records']].drop_duplicates().groupby('n_records', group_keys=False).apply(lambda x: x.sample(frac=.5, random_state=seed)).sid
        train_sid_type_df_dict[sid_type] = sid_type_df[sid_type_df.sid.isin(train_sids)]
        eval_sid_type_df_dict[sid_type] = sid_type_df[sid_type_df.sid.isin(train_sids) == False]
    
    return train_sid_type_df_dict, eval_sid_type_df_dict    


def sample_records(sid_type, sid_type_df, n_records, n_sids_to_sample, target_share_exact, seed, train_sid_list, eval_sid_list):
    
    sid_type_df = sid_type_df.copy()
    
    pool = sid_type_df[sid_type_df.n_records == n_records]['sid'].drop_duplicates().copy()
                    
    if (len(pool) == 0):
        return None

    if sid_type == 'singletons':
        if n_records == 1:
            n_sids_to_sample_this_pool = n_sids_to_sample
            logging.info(f"(SING) pool: {len(pool)}; n sids needed: {n_sids_to_sample_this_pool}")
    elif sid_type == 'non_singleton_em': 
        n_sids_to_sample_this_pool = int(np.round(target_share_exact * (n_sids_to_sample)))
        logging.info(f"(EM) pool: {len(pool)}; n sids needed: {n_sids_to_sample_this_pool}")
    elif sid_type == 'non_singleton_nem': 
        n_sids_to_sample_this_pool = int(np.round((1 - target_share_exact) * (n_sids_to_sample)))
        logging.info(f"(NEM) pool: {len(pool)}; n sids needed: {n_sids_to_sample_this_pool}")
    else:
        raise ValueError("Issue.")

    if (n_sids_to_sample_this_pool == 0):
        return None

    if n_sids_to_sample_this_pool > len(pool):
        
        sids = pool.tolist()
        target_n_sids = n_sids_to_sample_this_pool 
        while len(sids) < target_n_sids:
            n_sids_to_sample_this_pool = target_n_sids - len(sids)
            n_total = n_sids_to_sample_this_pool * n_records
            n_records = n_records - 1 
            n_sids_to_sample_this_pool = int(np.ceil(n_total / n_records))
            # NOTE: above is to account for fact that 3 13s isn't the same as 3 12s in terms of # of records
            pool = sid_type_df[(sid_type_df.n_records == n_records) & 
                            (sid_type_df.sid.isin(train_sid_list) == False) & 
                            (sid_type_df.sid.isin(eval_sid_list) == False)]['sid'].drop_duplicates().copy()
            logging.info(f"- triggered: {n_records}, need {n_sids_to_sample_this_pool}, have {len(pool)}")
            sids.extend(pool.sample(n=np.min([len(pool), n_sids_to_sample_this_pool]), random_state=seed))
        logging.info(f"- final size of last non-empty pool: {len(pool) - n_sids_to_sample_this_pool}")
        sids = pd.Series(sids)

    else: 
        sids = pool.sample(n=n_sids_to_sample_this_pool, random_state=seed)
        
    return sids


def get_administrative_datasets(df, train_sid_type_df_dict, eval_sid_type_df_dict, params):

    df = df.copy()

    seed = params['seed']
    exp_n = params['experiment_n']
    sample_size_list = params['sample_size_list']
    share_exact_list = params['share_exact_list']
    distribution_to_match = params.get('distribution_to_match__sids_by_cl_size', None)
    
    if distribution_to_match is not None:
        dist_sum = np.sum(list(distribution_to_match.values()))
        if dist_sum < .9999:
            raise ValueError(f"Distribution to target does not sum to 1. Sum is {dist_sum}.")
        n_sids_by_cl_size_to_match = {}
        for cl_size, sid_share in distribution_to_match.items():
            n_sids_by_cl_size_to_match[cl_size] = int(sid_share * df.sid.nunique())
    else: 
        target_share_singletons = share_singletons(df)
        n_sids_by_cl_size_to_match = df.groupby('sid').size().value_counts().to_dict()
        distribution_to_match = df.groupby('sid').size().value_counts(normalize=True).round(3).to_dict()
    
    logging.info("\nDistribution we're matching")
    logging.info(distribution_to_match)
    logging.info(n_sids_by_cl_size_to_match)

    total_n = np.sum([k*v for k, v in n_sids_by_cl_size_to_match.items()])

    train_df_dict = {}
    eval_df_dict = {}

    for target_n in sample_size_list:

        target_n_sids = 0
        for n_records, n_sids in n_sids_by_cl_size_to_match.items(): 
            target_n_as_pct = target_n / total_n
            n_records_to_sample = int(np.round(target_n_as_pct * n_sids * n_records))
            n_sids_to_sample = int(np.round(n_records_to_sample / n_records))
            target_n_sids += n_sids_to_sample
        
        # adjust the distribution to account for the experiment records we'll be taking out
        extras_we_need = exp_n / 2
        target_n_sids_per_cl_size = (pd.Series(distribution_to_match) * target_n_sids).to_dict()
        for n_records, share in distribution_to_match.items():
            extras_this_cl_size = int(extras_we_need * share)
            if n_records < np.max(list(distribution_to_match.keys())):
                target_n_sids_per_cl_size[n_records] = target_n_sids_per_cl_size[n_records] - extras_this_cl_size
                target_n_sids_per_cl_size[n_records + 1] = target_n_sids_per_cl_size[n_records + 1] + extras_this_cl_size
        adj_distribution_to_match = pd.Series(target_n_sids_per_cl_size)
        adj_distribution_to_match = adj_distribution_to_match/adj_distribution_to_match.sum()
        adj_n_sids_by_cl_size_to_match = {}
        for cl_size, sid_share in adj_distribution_to_match.to_dict().items():
            adj_n_sids_by_cl_size_to_match[cl_size] = int(sid_share * df.sid.nunique())

        total_n = np.sum([k*v for k, v in adj_n_sids_by_cl_size_to_match.items()])

        for target_share_exact in share_exact_list:
            
            logging.info(f"--- N: {target_n}; SHARE EXACT: {target_share_exact} ---")
        
            train_sid_list = []
            eval_sid_list = []

            for n_records, n_sids in adj_n_sids_by_cl_size_to_match.items(): 

                actual_target_n = target_n + extras_we_need
                n_as_pct = actual_target_n / total_n

                logging.info(f"n_records: {n_records}")
                n_records_to_sample = int(np.round(n_as_pct * n_sids * n_records))
                n_sids_to_sample = int(np.round(n_records_to_sample / n_records))
                
                for sid_type, sid_type_df in train_sid_type_df_dict.items(): 
                    train_sids = sample_records(sid_type, sid_type_df, n_records, 
                                                n_sids_to_sample, target_share_exact, seed, 
                                                train_sid_list, eval_sid_list)
                    if train_sids is None: 
                        continue
                    train_sid_list.extend(train_sids)
                    
                for sid_type, sid_type_df in eval_sid_type_df_dict.items(): 
                    eval_sids = sample_records(sid_type, sid_type_df, n_records, 
                                               n_sids_to_sample, target_share_exact, seed, 
                                               train_sid_list, eval_sid_list)
                    if eval_sids is None: 
                        continue
                    eval_sid_list.extend(eval_sids)

            train_df_dict[f"{target_n}__{int(target_share_exact*100)}"] = df[df.sid.isin(train_sid_list)].copy()
            eval_df_dict[f"{target_n}__{int(target_share_exact*100)}"] =  df[df.sid.isin(eval_sid_list)].copy()

    return train_df_dict, eval_df_dict


def get_experiment_datasets(full_pool_df, admin_df_dict, elig_singletons, exp_n, seed):

    full_pool_df = full_pool_df.copy()
    admin_df_dict = admin_df_dict.copy()

    exp_df_dict = {}
    updated_admin_df_dict = {}

    for specification, admin_df in admin_df_dict.items():

        target_share_exact = float(specification.split('__')[1])/100
        n_to_extract = int(exp_n / 2)
        n_em_to_extract = int(n_to_extract * target_share_exact)
        n_nem_to_extract = int(n_to_extract * (1 - target_share_exact))
        
        exp_rowid_list = elig_singletons.sample(n_to_extract, random_state=seed).rowid.tolist()
        
        em_rows = (
            admin_df[(admin_df.n_records > 1) & (admin_df.any_non_exact_match == 0)]
            .drop_duplicates(subset=['sid'])
            .sample(n_em_to_extract, random_state=seed).rowid.tolist())
        
        nem_rows = (
            admin_df[(admin_df.n_records > 1) & (admin_df.all_non_exact_match == 1)]
            .drop_duplicates(subset=['sid'])
            .sample(n_nem_to_extract, random_state=seed).rowid.tolist())
        
        exp_rowid_list.extend(em_rows)
        exp_rowid_list.extend(nem_rows)
        
        updated_admin_df_dict[specification] = admin_df[admin_df.rowid.isin(exp_rowid_list) == False].copy()
        exp_df_dict[specification] = full_pool_df[full_pool_df.rowid.isin(exp_rowid_list)].copy()

    return updated_admin_df_dict, exp_df_dict


def get_datasets(df, params):

    df = df.copy()
    
    sid_type_df_dict = {
        'singletons': df[df.n_records == 1].copy(),
        'non_singleton_em': df[(df.n_records > 1) & (df.any_non_exact_match == 0)].copy(),
        'non_singleton_nem': df[(df.n_records > 1) & (df.all_non_exact_match == 1)].copy()
    }
    
    train_sid_type_df_dict, eval_sid_type_df_dict = train_eval_split(sid_type_df_dict, params['seed'])

    train_df_dict, eval_df_dict = get_administrative_datasets(
        df, 
        train_sid_type_df_dict, 
        eval_sid_type_df_dict,
        params)

    all_training_sids = list(pd.concat(list(train_df_dict.values())).sid.unique())
    all_eval_sids = list(pd.concat(list(eval_df_dict.values())).sid.unique())
    singletons = sid_type_df_dict['singletons'].copy()
    elig_singletons = singletons[(singletons.sid.isin(all_eval_sids) == False) & 
                                 (singletons.sid.isin(all_training_sids) == False)].copy()    

    admin_train_df_dict, exp_train_df_dict = get_experiment_datasets(
        df,
        train_df_dict, 
        elig_singletons, 
        params['experiment_n'],
        params['seed'])
        
    admin_eval_df_dict, exp_eval_df_dict = get_experiment_datasets(
        df,
        eval_df_dict, 
        elig_singletons, 
        params['experiment_n'],
        params['seed'])

    return admin_train_df_dict, admin_eval_df_dict, exp_train_df_dict, exp_eval_df_dict


def create_eligible_sampling_pool(df, params):

    df = df.copy()

    df = add_stat_columns(df)

    singletons = df[df.n_records == 1].copy()
    non_singletons = df[(df.n_records > 1)].copy()
    non_singletons_em = df[(df.n_records > 1) & (df.n_uq_namedob == 1)].copy()
    non_singletons_nem = df[(df.n_records > 1) & (df.n_uq_namedob == df.n_records)].copy()
    non_singletons_mixed = df[(df.n_records > 1) & (df.n_uq_namedob != 1) & (df.n_uq_namedob != df.n_records)].copy()
    non_singletons_mixed_collapsed = df[
            (df.n_records > 1) & (df.n_uq_namedob != 1) & (df.n_uq_namedob != df.n_records)
            ].drop_duplicates(subset=['sid', 'first_name', 'last_name', 'date_of_birth'])

    # recalculate stats for the collapsed version 
    non_singletons_mixed_collapsed = add_stat_columns(non_singletons_mixed_collapsed)
    
    all_non_singletons_nem = pd.concat([
        non_singletons_nem,
        non_singletons_mixed_collapsed
    ])
    
    non_singletons_em, all_non_singletons_nem = truncate_large_clusters(
        non_singletons_em, all_non_singletons_nem, params)

    full_pool_df = pd.concat([
        singletons,
        non_singletons_em, 
        all_non_singletons_nem
    ])
    
    full_pool_df = add_stat_columns(full_pool_df)

    return full_pool_df


def truncate_large_clusters(non_singletons_em, all_non_singletons_nem, params):
    
    max_cl_size = params.get('max_cluster_size', None)
    thresh = params.get('max_cluster_size_pctl_thresh', None)
    drop_em = params.get('skip_reallocation_of_big_em_clusters', True)
    bins_to_dist_nem = params.get('n_bins_to_reallocate_to_nem', 2)
    bins_to_dist_em = params.get('n_bins_to_reallocate_to_em', 4)
    seed = params['seed']
    
    non_singletons_em = non_singletons_em.copy()
    all_non_singletons_nem = all_non_singletons_nem.copy()
    
    if max_cl_size is None:
        if thresh is not None: 
            nem_cl_sizes = all_non_singletons_nem.groupby('sid').size().value_counts(normalize=True).sort_index()
            nem_cl_sizes = nem_cl_sizes.cumsum()
            max_cl_size = nem_cl_sizes[nem_cl_sizes < thresh].tail(1).index[0]
        else: 
            return non_singletons_em, all_non_singletons_nem
    
    if drop_em:
        small_enough_clusters = non_singletons_em.groupby('sid').size()
        small_enough_clusters = small_enough_clusters[small_enough_clusters <= max_cl_size].index
        non_singletons_em = non_singletons_em[non_singletons_em.sid.isin(small_enough_clusters)]
    else:
        non_singletons_em = _truncate_large_clusters(non_singletons_em, max_cl_size, bins_to_dist_em, seed)
        
    all_non_singletons_nem = _truncate_large_clusters(all_non_singletons_nem, max_cl_size, bins_to_dist_nem, seed)
    
    return non_singletons_em, all_non_singletons_nem


def _truncate_large_clusters(df, max_cl_size, bins_to_dist=2, seed=3):

    df = df.copy()
    
    cl_size = df.groupby('sid').size()
    too_big_clusters = pd.Series(cl_size[cl_size > max_cl_size].index)
    
    bins_to_dist = [max_cl_size - (i) for i in np.arange(bins_to_dist)]
    
    cl_size_dist = cl_size.value_counts(normalize=True)
    bins_to_dist_dist = cl_size_dist[cl_size_dist.index.isin(bins_to_dist)]
    bins_to_dist_dist = bins_to_dist_dist/bins_to_dist_dist.sum()
    
    adj_df = df[df.sid.isin(too_big_clusters) == False].copy()
    
    already_truncated = []
    for target_cl_size, share in bins_to_dist_dist.to_dict().items():
        
        elig_too_big_clusters = too_big_clusters[too_big_clusters.isin(already_truncated) == False]
        this_pool_too_big_clusters = elig_too_big_clusters.sample(n=int(np.round(len(too_big_clusters)*share)), random_state=seed)
        
        this_pool_adj_df = df[df.sid.isin(this_pool_too_big_clusters)].groupby('sid').apply(lambda x: x.sample(n=target_cl_size, random_state=seed))
        
        adj_df = pd.concat([adj_df, this_pool_adj_df])
        
        already_truncated.extend(this_pool_too_big_clusters)

    assert(df.sid.nunique() == adj_df.sid.nunique())
    
    return adj_df

def output_datasets(df_dict, dataset_type, output_dir):

    df_dict = df_dict.copy()

    cols_to_drop = [
        'name_dob', 'sid_name_dob', 'n_records', 'n_uq_namedob', 
        'n_records_per_sid_name_dob', 'any_non_exact_match', 'all_non_exact_match']

    for specification, df in df_dict.items():
        df = df.drop(columns=cols_to_drop)
        df['dataset'] = dataset_type
        if dataset_type != 'admin_training':
            df['uid_sid'] = np.NaN
        output_file = os.path.join(output_dir, f"{dataset_type}__{specification}.csv")
        df.to_csv(output_file, index=False)


def main(args):

    params = yaml.safe_load(open(args.config_file,'r'))
    df = pd.read_csv(args.data_file)
    
    # clean data and generate samples
    df = preprocess_data(df, params['col_with_lineage'])
    df = create_eligible_sampling_pool(df, params)
    admin_train_df_dict, admin_eval_df_dict, exp_train_df_dict, exp_eval_df_dict = get_datasets(df, params)

    # output 
    output_datasets(admin_train_df_dict, 'admin_training', args.output_dir)
    output_datasets(admin_eval_df_dict, 'admin_evaluation', args.output_dir)
    output_datasets(exp_train_df_dict, 'experiment_training', args.output_dir)
    output_datasets(exp_eval_df_dict, 'experiment_evaluation', args.output_dir)
    

    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file')
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
