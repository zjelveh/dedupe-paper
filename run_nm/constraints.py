
import numpy as np
import pandas as pd


def get_columns_used():

    an_dtypes = {
        'dataset':str
    }
    
    return an_dtypes


def is_valid_link(predicted_links_df):

    predicted_links_df['valid'] = True

    # don't allow links between admin_training and anything else
    predicted_links_df.loc[
        (predicted_links_df.dataset_1 == 'ADMIN_TRAINING') & (predicted_links_df.dataset_2 != 'ADMIN_TRAINING'),
        'valid'
    ] = False
    predicted_links_df.loc[
        (predicted_links_df.dataset_2 == 'ADMIN_TRAINING') & (predicted_links_df.dataset_1 != 'ADMIN_TRAINING'),
        'valid'
    ] = False

    # prohibit links between two experiment records
    predicted_links_df.loc[
        (predicted_links_df.dataset_1 == 'EXPERIMENT') & (predicted_links_df.dataset_2 == 'EXPERIMENT'),
        'valid'
    ] = False

    return predicted_links_df['valid']
    

def is_valid_cluster(cluster, phat=None):

    # at most one experiment record per cluster
    if (cluster.dataset == 'EXPERIMENT').sum() > 1:
        return False

    # clusters cannot contain an admin_training record and a record from another dataset
    # Note: this might not be possible given the above edge constraint
    if (cluster.dataset == 'ADMIN_TRAINING').sum() > 0: 
        if (cluster.dataset != 'ADMIN_TRAINING').sum() > 0: 
            return False

    return True


def apply_link_priority(valid_links_df):

    valid_links_df.sort_values(by=['phat', 'original_order'], ascending=[False, True])

    return valid_links_df
