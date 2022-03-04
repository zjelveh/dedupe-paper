
import pandas as pd
import numpy as np
import os

import data_viz_tools as cel_viz
import matplotlib.pyplot as plt
import seaborn as sns
cel_viz.set_plot_configs(kind='presentation', rc={'figure.dpi': 150})
from matplotlib.ticker import StrMethodFormatter

recall_ylim = (.35, 1.01)
precision_ylim = (.9, 1.01)
error_ylim = (0, .65)

default_recall_ylim = (-.05, 1.01)
default_precision_ylim = (.9, 1.01)
default_error_ylim = (0, 1.05)

from collections import defaultdict
alg_color_map = defaultdict(lambda x: 'grey')
alg_color_map['namematch'] = 'black'
alg_color_map['Deduplication'] = '#01579b'
alg_color_map['Record Linkage'] = '#1eb2f5'


def get_main_result_plots(data_df, any_or_corr='corr', all_or_nem='all', metrics=['tpr', 'prc', 'error'], 
                          ci=False, nm_lines='median', draw_nm_region=False, min_runtime=True, legend=True, sample_desc='all_specifications', 
                          savefig=False, force_ylim=None, figure_output_dir=''):
    
    hue_order = ['Deduplication True', 'Deduplication False', 'Record Linkage True', 'Record Linkage False']
    
    alg_requirement_marker_map = {
        'Deduplication True': '$\\bullet$',
        'Record Linkage True': '$\\bullet$',
        'Deduplication False': '$\\circ$',
        'Record Linkage False': '$\\circ$'
    }
    
    alg_requirement_color_map = {
        'Deduplication True': alg_color_map['Deduplication'],
        'Record Linkage True': alg_color_map['Record Linkage'],
        'Deduplication False': alg_color_map['Deduplication'],
        'Record Linkage False': alg_color_map['Record Linkage']
    }
    
    suffix = '' if all_or_nem == 'all' else '__nem'
    if all_or_nem is not None:
        metrics = [f"{m}_{any_or_corr}{suffix}" for m in metrics]
        metrics = [m.replace('error_corr', 'error_pair') for m in metrics]
    
    for metric in metrics:
        
        plot_df = data_df[data_df.algorithm.str.contains('dedupe')].copy()
        if min_runtime and ('runtime' in metric):
            plot_df = plot_df.groupby(['budget', 'framework', 'N Admin Rows'])[metric].min().reset_index()
            
        plot_df['meets_label_requirements'] = True
        try:
            plot_df.loc[(plot_df.labeled_1s < 10) | (plot_df.labeled_0s < 10), 'meets_label_requirements'] = False
        except:
            pass
        
        plot_df['framework__meets_label_requirements'] = plot_df.framework + ' ' + plot_df.meets_label_requirements.astype(str)

        budget_map = {20:1, 40: 2, 80:3, 200:4, 500:5, 1000:6}
        plot_df['budget_map'] = plot_df['budget'].map(budget_map)
        
        g = sns.lmplot(
            data=plot_df,
            x='budget_map',
            y=metric,
            hue='framework__meets_label_requirements',
            col='N Admin Rows',
            legend=False,
            aspect=0.9,
            ci=False,
            fit_reg=False,
            hue_order=hue_order,
            palette=sns.color_palette([alg_requirement_color_map[alg] for alg in hue_order]),
            markers=[alg_requirement_marker_map[m] for m in hue_order],
            scatter_kws={'s':150, 'linewidth':.001}
        )
        
        _ = plt.xlim(0.5, 6.5)
        if 'tpr' in metric:
            metric_str = 'Recall'
        elif 'prc' in metric: 
            metric_str = 'Precision'
        elif 'error' in metric:
            metric_str = 'Total Error'
        elif metric == 'runtime_min': 
            metric_str = 'Runtime (minutes)'
        else: 
            metric_str = metric.capitalize()
        
        if metric_str == 'Precision':
            _ = plt.ylim(.9, 1.01)
        if force_ylim == 'best':
            if metric_str == 'Precision':
                _ = plt.ylim(precision_ylim[0], precision_ylim[1])
            if metric_str == 'Recall':
                _ = plt.ylim(recall_ylim[0], recall_ylim[1])
            if metric_str == 'Total Error':
                _ = plt.ylim(error_ylim[0], error_ylim[1])
        elif force_ylim == 'default':
            if metric_str == 'Precision':
                _ = plt.ylim(default_precision_ylim[0], default_precision_ylim[1])
            if metric_str == 'Recall':
                _ = plt.ylim(default_recall_ylim[0], default_recall_ylim[1])
            if metric_str == 'Total Error':
                _ = plt.ylim(default_error_ylim[0], default_error_ylim[1])
        
        _ = plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            
        _ = g.set_ylabels(metric_str)
        _ = g.set_xlabels('Budget')

        if legend:
            _ = g.add_legend(
                title='', fontsize='large', 
                legend_data={k.replace(' True', ''): v for k, v in g._legend_data.items() if k in list(plot_df.framework__meets_label_requirements) and 'True' in k},
                label_order =[k.replace(' True', '') for k in list(g._legend_data.keys()) if 'True' in k]
            )
        
        for ax_row in g.axes:
            for ax in ax_row:
                conditions_dict = dict(zip([x.split(' = ')[0].replace("'", '') for x in ax.title._text.split(' | ')],
                                           [int(x.split(' = ')[1].replace("'", '')) for x in ax.title._text.split(' | ')]))
                nm_val_df = data_df[(data_df.algorithm == 'namematch')].copy()
                for condition_col, condition_val in conditions_dict.items():
                    nm_val_df = nm_val_df[(nm_val_df[condition_col] == condition_val)]
                if nm_lines == 'mean': 
                    nm_val_df = nm_val_df.groupby(['algorithm'])[metric].mean().reset_index()
                elif nm_lines == 'median': 
                    nm_val_df['dist_from_50'] = (nm_val_df.groupby(['algorithm'])[metric].rank(pct=True) - .5).abs()
                    nm_val_df['min_dist_from_50'] = nm_val_df.groupby(['algorithm'])['dist_from_50'].transform('min')
                    nm_val_df = nm_val_df[nm_val_df.min_dist_from_50 == nm_val_df.dist_from_50].groupby(['algorithm']).head(1)
                elif 'first' in nm_lines:
                    nm_val_df = nm_val_df[nm_val_df.iter == 0]
                for r in np.arange(len(nm_val_df)):
                    if (nm_lines == 'median') and ('runtime' not in metric):
                        nm_val = nm_val_df[f"median__{metric}"].iloc[r]
                    else:
                        if nm_lines == 'point_est_of_first':
                            nm_val = nm_val_df[metric].iloc[r]
                        elif nm_lines == 'median_of_first':
                            nm_val = nm_val_df[f"median__{metric}"].iloc[r]
                        elif nm_lines == 'median':
                            nm_val = nm_val_df[metric].iloc[r]
                    alg = nm_val_df['algorithm'].iloc[r]
                    l = ax.axhline(nm_val, ls='--', linewidth=1.5, color=alg_color_map[alg])
                    if draw_nm_region and ('runtime' not in metric):
                        nm_val_error_min = nm_val_df[f"lower_bound__{metric}"].iloc[r]
                        nm_val_error_max = nm_val_df[f"upper_bound__{metric}"].iloc[r]
                        _ = ax.axhspan(nm_val_error_min, nm_val_error_max, color='#d4d4d4', alpha=0.3)
                        
        for ax_row in g.axes:
            for ax in ax_row:
                _ = ax.set_xticks(ticks=np.arange(1, 7, 1))
                _ = g.set_xticklabels(labels=[20, 40, 80, 200, 500, 1000])
                _ = ax.grid(False, axis='x')
                
        plt.show()
        
        if savefig:
            if draw_nm_region:
                filepath = os.path.join(figure_output_dir, f'main_result__{sample_desc}__{metric}__with_grey.png')
            else:
                filepath = os.path.join(figure_output_dir, f'main_result__{sample_desc}__{metric}.png')
            g.savefig(filepath)


def get_sample_size_plots(data_df, any_or_corr='corr', all_or_nem='all', metrics=['tpr', 'prc', 'error'], 
                          ci=False, avg_nm=True, min_runtime=True, drop_record_linkage=False, 
                          sample_desc='all_specifications', savefig=False, figure_output_dir=''):
    
    hue_order = ['Deduplication (1500)', 'Deduplication (15000)', 'Deduplication (150000)', 'Deduplication (300000)', 
                 'Record Linkage (1500)', 'Record Linkage (15000)', 'Record Linkage (150000)', 'Record Linkage (300000)']
    if drop_record_linkage:
        hue_order = [h for h in hue_order if'Record' not in h]
        
    marker_map = {
        'Deduplication (1500)': '$\\circ$', 
        'Deduplication (150000)': '$\\bullet$', 
        'Deduplication (15000)': '$\\times$', 
        'Deduplication (300000)': '$\\ast$', 
        'Record Linkage (1500)': '$\\circ$', 
        'Record Linkage (150000)': '$\\bullet$', 
        'Record Linkage (15000)': '$\\times$', 
        'Record Linkage (300000)': '$\\ast$'
    }
    
    suffix = '' if all_or_nem == 'all' else '__nem'
    if all_or_nem is not None:
        metrics = [f"{m}_{any_or_corr}{suffix}" for m in metrics]
        metrics = [m.replace('error_corr', 'error_pair') for m in metrics]
    
    for metric in metrics:
        
        plot_df = data_df[data_df.algorithm.str.contains('dedupe')].copy()
        if min_runtime and ('runtime' in metric):
            plot_df = plot_df.groupby(['budget', 'framework', 'N Admin Rows'])[metric].min().reset_index()
        if drop_record_linkage:
            plot_df = plot_df[plot_df.framework.str.contains('Deduplication')]
        
        if 'error_pair' in metric:
            display(Markdown('<font color="red">NOTE: pair level, not "corr"</font>'))
            
        plot_df = plot_df.groupby(['budget', 'N Admin Rows', 'framework'])[metric].mean().reset_index()
        
        budget_map = {20:1, 40: 2, 80:3, 200:4, 500:5, 1000:6}
        plot_df['budget_map'] = plot_df['budget'].map(budget_map)
            
        g = sns.lmplot(
            data=plot_df,
            x='budget_map',
            y=metric,
            hue='framework',
            col='N Admin Rows',
            legend=False,
            aspect=0.9,
            ci=False,
            fit_reg=False,
            hue_order=hue_order,
            #palette=sns.color_palette([alg_requirement_color_map[alg] for alg in hue_order]),
            markers=[marker_map[m] for m in hue_order],
            scatter_kws={'s':150}
        )
        
        if 'tpr' in metric:
            metric_str = 'Recall'
        elif 'prc' in metric: 
            metric_str = 'Precision'
        elif 'error' in metric:
            metric_str = 'Total Error'
        elif metric == 'runtime_min': 
            metric_str = 'Runtime (minutes)'
        else: 
            metric_str = metric.capitalize()
            
        if metric_str == 'Precision':
            _ = plt.ylim(.9, 1.01)
            
        _ = plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            
        _ = g.set_ylabels(metric_str)
        _ = g.set_xlabels('Budget')
        _ = g.add_legend(title='', fontsize='large')
        
        for ax_row in g.axes:
            for ax in ax_row:
                _ = ax.set_xticks(ticks=np.arange(1, 7, 1))
                _ = g.set_xticklabels(labels=[20, 40, 80, 200, 500, 1000])
                _ = ax.grid(False, axis='x')
        
        plt.show()
        
        if savefig:
            filepath = os.path.join(figure_output_dir, f'sample_size_result__{sample_desc}__{metric}.png')
            g.savefig(filepath)