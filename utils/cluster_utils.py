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
    