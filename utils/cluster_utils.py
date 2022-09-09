import pandas as pd
import numpy as np

def cluster_edges(clustered_dupes):
    cluster_assignments = {}
    clusters = {}

    for potential_cluster_id, (records, scores) in enumerate(clustered_dupes):
        record_id_1 = records[0]
        record_id_2 = records[1]
        new_cluster = [record_id_1, record_id_2]

        if record_id_1 in cluster_assignments and record_id_2 in cluster_assignments:
            if cluster_assignments[record_id_1] == cluster_assignments[record_id_2]:
                continue

        if record_id_1 in cluster_assignments:
            old_cluster_id_1 = cluster_assignments[record_id_1]
            old_cluster_id_1_records = clusters[old_cluster_id_1]
            cluster_id = old_cluster_id_1
            new_cluster.extend(old_cluster_id_1_records)
            _ = clusters.pop(old_cluster_id_1)

            if record_id_2 in cluster_assignments:
                old_cluster_id_2 = cluster_assignments[record_id_2]
                old_cluster_id_2_records = clusters[old_cluster_id_2]
                new_cluster.extend(old_cluster_id_2_records)
                _ = clusters.pop(old_cluster_id_2)

        elif record_id_2 in cluster_assignments:
            old_cluster_id_2 = cluster_assignments[record_id_2]
            old_cluster_id_2_records = clusters[old_cluster_id_2]
            cluster_id = old_cluster_id_2
            new_cluster.extend(old_cluster_id_2_records)
            _ = clusters.pop(old_cluster_id_2)

        else: 
            cluster_id = potential_cluster_id

        new_cluster = list(set(new_cluster))
        clusters[cluster_id] = new_cluster
        for record_id in new_cluster:
            cluster_assignments[record_id] = cluster_id
            
    return cluster_assignments


def convert_predictions_to_all_names_with_clusterid(exp, admin, predicted_matches=None, predicted_clusters=None):

    if (predicted_matches is None) + (predicted_clusters is None) != 1:
        raise ValueError("Only one of predicted_matches and predicted_clusters should be non-None.")

    # cluster
    if predicted_clusters is None:
        edges = []
        for row_i, match_row in predicted_matches.iterrows():
            edges.append([(match_row['rowid1'], match_row['rowid2']), 1]) # fake score of 1 to match dedupe format
        cluster_membership = cluster_edges(edges)

        # get rowid, cluster_id mapping
        predicted_clusters = pd.DataFrame.from_dict(cluster_membership, orient='index')
        predicted_clusters = predicted_clusters.reset_index()
        predicted_clusters.columns = ['rowid', 'cluster_id']

    # stack and merge to get sid
    an_w_cluster = pd.concat([exp, admin])[['rowid', 'sid', 'dataset', 'date_of_birth']]
    len_before = len(an_w_cluster)
    an_w_cluster = an_w_cluster.merge(predicted_clusters, on='rowid', how='left')
    len_after = len(an_w_cluster)
    assert len_before == len_after

    # fillna cluster ids
    an_w_cluster.loc[(an_w_cluster.dataset == 'admin_evaluation') & (an_w_cluster.cluster_id.isna()), 'cluster_id'] = -2
    an_w_cluster.loc[(an_w_cluster.dataset == 'experiment_evaluation') & (an_w_cluster.cluster_id.isna()), 'cluster_id'] = -1
    an_w_cluster['cluster_id'] = an_w_cluster.cluster_id.astype(int)

    return an_w_cluster
