import pandas as pd
import numpy as np
import argparse
import joblib
import yaml
import os
import sys


def main(args):

    config_template = yaml.safe_load(open(args.config_template_file,'r'))
    config_template['data_files']['experiment']['filepath'] = args.experiment_file
    config_template['data_files']['admin_training']['filepath'] = args.training_file
    config_template['data_files']['admin_evaluation']['filepath'] = args.evaluation_file

    if args.task == 'ojin':
        config_template['blocking_thresholds'] = {
            'common_name_max_penalty' : 0.05,   # 0.10 default
            'nodob_cosine_bar' :  0.36,         # 0.26 default (this one doesn't matter here)
            'high_cosine_bar' : 0.55,           # 0.30 default
            'low_cosine_bar' : 0.55,            # 0.40 default
            'high_editdist_bar' : 3,            # 1    default
            'low_editdist_bar' : 3,             # 2    default
            'absvalue_bar' : 3                  # 3    default (this one doesn't matter here)
        } 
        config_template['blocking_scheme'] = {
            'cosine_distance' : {
                'variables' : ['first_name', 'last_name']
            },
            'edit_distance' : {
                'variable' : 'dob'
            },
            'absvalue_distance' :  { # serves as a backup second filter (e.g. if no dob)
                'variable' : 'age'
            },
            'alpha' : 1.2,
            'power' : 0.1
        }

    yaml_file = os.path.join(args.specification_dir, 'config.yaml')
    with open(yaml_file, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False)

    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_template_file')
    parser.add_argument('--specification_dir')
    parser.add_argument('--task')
    parser.add_argument('--experiment_file')
    parser.add_argument('--training_file')
    parser.add_argument('--evaluation_file')
    args = parser.parse_args()  

    main(args)