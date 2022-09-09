
import pandas as pd
import numpy as np
import os
import warnings

import ul_py_viz
import matplotlib.pyplot as plt
import seaborn as sns
ul_py_viz.set_plot_configs(kind='presentation', rc={'figure.dpi': 150})
from matplotlib.ticker import StrMethodFormatter

from collections import defaultdict

import string
alphabet_list = list(string.ascii_uppercase)


ylim_dict = {
    'Recall': {
        'deduplication': {'default': (-.05, 1.01), 'optimized': (.33, 1.01)}, 
        'record_linkage': {'default': (-.05, 1.01), 'optimized': (.33, 1.01)}},
    'Precision': {
        'deduplication': {'default': (.85, 1.01), 'optimized': (.85, 1.01)}, 
        'record_linkage': {'default': (.73, 1.01), 'optimized': (.73, 1.01)}},
    'Total Error': {
        'deduplication': {'default': (0, 1.05), 'optimized': (0, .70)},
        'record_linkage': {'default': (0, 1.05), 'optimized': (0, .70)}}
}

alg_color_map = defaultdict(lambda x: 'grey')
alg_color_map['namematch'] = 'black'
alg_color_map['Deduplication'] = '#01579b'
alg_color_map['Record Linkage'] = '#1eb2f5'


def get_main_result_plots(data_df, any_or_corr='corr', metrics=['tpr', 'prc', 'error'],
                          significance_metric='error_any',
                          benchmark_lines='point_est_of_first', benchmark_uncertainty_shading=False,
                          legend=True, force_ylim=None, sample_desc='all_specifications',
                          savefig=False, figure_output_dir=''):
    '''
    NOTE: nm_lines and fl_lines can take values: mean, median, first, all
    '''
    
    # MAPPINGS NEEDED FOR COLORS, MARKERS, AND X-VALUES
    
    hue_order = ['deduplication True', 'deduplication False', 'record_linkage True', 'record_linkage False']
    
    alg_requirement_marker_map = {
        'deduplication True': '$\\bullet$',
        'record_linkage True': '$\\bullet$',
        'deduplication False': '$\\circ$',
        'record_linkage False': '$\\circ$'
    }
    
    alg_requirement_color_map = {
        'deduplication True': alg_color_map['Deduplication'],
        'record_linkage True': alg_color_map['Record Linkage'],
        'deduplication False': alg_color_map['Deduplication'],
        'record_linkage False': alg_color_map['Record Linkage']
    }
    
    budget_map = {20:1, 40: 2, 80:3, 200:4, 500:5, 1000:6}
    
    # GET ACTUAL METRIC COLUMNS BASED ON ARGUMENTS
    
    metrics = [f"{m}_{any_or_corr}" for m in metrics]
    metrics = [m.replace('error_corr', 'error_pair') for m in metrics]
    
    # CREATE THE PLOT DF FOR DEDUPE RESULTS
    
    dedupe_plot_df = data_df[data_df.algorithm.str.contains('dedupe')].copy()
    dedupe_plot_df['meets_label_requirements'] = True
    dedupe_plot_df.loc[(dedupe_plot_df.labeled_1s < 10) | (dedupe_plot_df.labeled_0s < 10), 'meets_label_requirements'] = False
    dedupe_plot_df['framework__meets_label_requirements'] = dedupe_plot_df.framework + ' ' + dedupe_plot_df.meets_label_requirements.astype(str)
    dedupe_plot_df['budget_map'] = dedupe_plot_df['budget'].map(budget_map)
    
    # CREATE THE PLOT
    
    for metric_i, metric in enumerate(metrics):
        
        g = sns.lmplot(
            data=dedupe_plot_df,
            x='budget_map',
            y=metric,
            hue='framework__meets_label_requirements',
            col='ss_train',
            legend=False,
            aspect=0.9,
            ci=False,
            fit_reg=False,
            hue_order=hue_order,
            palette=sns.color_palette([alg_requirement_color_map[alg] for alg in hue_order]),
            markers=[alg_requirement_marker_map[m] for m in hue_order],
            scatter_kws={'s':150, 'linewidth':.001}
        )
        
        if 'tpr' in metric:
            metric_str = 'Recall'
        elif 'prc' in metric: 
            metric_str = 'Precision'
        elif 'error' in metric:
            metric_str = 'Total Error'
        else: 
            metric_str = metric.capitalize()
            
        if force_ylim == 'best':
            dedupe_version = 'optimized'
        else:
            dedupe_version = 'default'
            
        if 'record_linkage' in sample_desc:
            framework = 'record_linkage'
        else:
            framework = 'deduplication'
            
        try:
            ylims = ylim_dict[metric_str][framework][dedupe_version]
            _ = plt.ylim(ylims[0], ylims[1])
        except:
            pass
            
        _ = plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            
        _ = g.set_ylabels(metric_str)
        _ = g.set_xlabels('Budget')

        if legend:
            _ = g.add_legend(
                title='', fontsize='large', 
                legend_data={k.replace(' True', ''): v for k, v in g._legend_data.items() 
                             if k in list(plot_df.framework__meets_label_requirements) and 'True' in k},
                label_order =[k.replace(' True', '') for k in list(g._legend_data.keys()) if 'True' in k]
            )
        
        for ax_row in g.axes:
            for ax in ax_row:
                
                conditions_dict = dict(zip([x.split(' = ')[0].replace("'", '') for x in ax.title._text.split(' | ')],
                                           [int(x.split(' = ')[1].replace("'", '')) for x in ax.title._text.split(' | ')]))
                
                for hline in ['fl', 'nm']:
                    
                    if benchmark_lines is None or benchmark_lines is False:
                        break
                    
                    if hline == 'nm':
                        algorithm = 'namematch'
                        color = '#6a6a6a'
                        region_color = '#d4d4d4'
                        line_style = '-'
                        label = 'Name Match'
                    elif hline == 'fl':
                        algorithm = 'fastlink (deduplication)'
                        if 'record_linkage' in sample_desc:
                            algorithm = 'fastlink (record_linkage)'
                        color= '#bf7f76'
                        region_color = '#f2ded6'
                        line_style = '--'
                        label = 'fastLink'
                    else:
                        raise ValueError("hline not defined")
                        
                    all_benchmark_df = data_df[(data_df.algorithm == algorithm)].copy()
                    for condition_col, condition_val in conditions_dict.items():
                        all_benchmark_df = all_benchmark_df[(all_benchmark_df[condition_col] == condition_val)]
                        
                    if benchmark_lines == 'mean': 
                        benchmark_df = all_benchmark_df[metric].mean().reset_index()
                    
                    elif benchmark_lines == 'median': 
                        benchmark_df = all_benchmark_df[all_benchmark_df.is_median == True]
                    
                    elif benchmark_lines == 'first':
                        benchmark_df = all_benchmark_df[all_benchmark_df.model_iter == all_benchmark_df.model_iter.min()]
                        
                    elif benchmark_lines == 'all':
                        benchmark_df = all_benchmark_df.copy()
                    
                    for r in np.arange(len(benchmark_df)):
                        hval = benchmark_df[metric].iloc[r]
                        l = ax.axhline(hval, ls=line_style, linewidth=1.5, color=color, label=label)
                        if benchmark_uncertainty_shading is not None and benchmark_uncertainty_shading is not False:
                            if benchmark_uncertainty_shading == 'median_only':
                                if not benchmark_df['is_median'].iloc[r]:
                                    continue
                            hval_error_min = benchmark_df[f"{metric}__lower_bound"].iloc[r]
                            hval_error_max = benchmark_df[f"{metric}__upper_bound"].iloc[r]
                            _ = ax.axhspan(hval_error_min, hval_error_max, color=region_color, alpha=0.3)
        
                _ = ax.set_xticks(ticks=np.arange(1, 7, 1))
                _ = g.set_xticklabels(labels=[20, 40, 80, 200, 500, 1000])
                _ = ax.grid(False, axis='x')
                
        _ = g.set_titles(col_template='N Admin Rows = {col_name}')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if metric_i == 1:
                if benchmark_uncertainty_shading:
                    # the underscores are a hack to hide some of the legend options we don't want to show
                    _ = plt.legend(labels=['Dedupe', "_dedupe_empty_circles", 'fastLink', '_fl_region', 'Name Match', '_nm_region'],
                                bbox_to_anchor=(1.05, .6), loc='upper left', borderaxespad=0, frameon=True)
                else:
                    # the underscores are a hack to hide some of the legend options we don't want to show
                    _ = plt.legend(labels=['Dedupe', "_dedupe_empty_circles", 'fastLink', 'Name Match'],
                                bbox_to_anchor=(1.05, .6), loc='upper left', borderaxespad=0, frameon=True)
            else:
                # hack so the legend only print on the middle of three plot "rows"
                _ = plt.legend(handles=[plt.plot([], ls="-", color='white')[0]], labels=['Name Match'],
                               bbox_to_anchor=(1.05, .6), loc='upper left', borderaxespad=0, frameon=False, labelcolor='white')
                
        _ = g.fig.subplots_adjust(top=.8, right=.92) 
        _ = g.fig.suptitle(f'({alphabet_list[metric_i]})', fontsize='x-large')

        plt.show()
        
        if savefig:
            if benchmark_uncertainty_shading:
                filepath = os.path.join(figure_output_dir, f'main_result__{sample_desc}__{metric}__with_shading.png')
            else:
                filepath = os.path.join(figure_output_dir, f'main_result__{sample_desc}__{metric}.png')
            g.savefig(filepath)


def get_sample_size_plots(data_df, any_or_corr='corr', metrics=['tpr', 'prc', 'error'], 
                          sample_desc='all_specifications', diff_markers=False, 
                          savefig=False, figure_output_dir=''):
    
    # MAPPINGS NEEDED FOR COLORS, MARKERS, AND X-VALUES
    
    hue_order = ['Sample size: 1,500', 'Sample size: 15,000', 'Sample size: 150,000', 'Sample size: 300,000']
        
    marker_map = {
        'Sample size: 1,500': '$\\circ$', 
        'Sample size: 15,000': '$\\bullet$', 
        'Sample size: 150,000': '$\\times$', 
        'Sample size: 300,000': '$\\ast$'
    }
    
    budget_map = {20:1, 40: 2, 80:3, 200:4, 500:5, 1000:6}
    
    # GET ACTUAL METRIC COLUMNS BASED ON ARGUMENTS
    
    metrics = [f"{m}_{any_or_corr}" for m in metrics]
    metrics = [m.replace('error_corr', 'error_pair') for m in metrics]
               
    # CREATE THE PLOT DF FOR DEDUPE RESULTS
    
    dedupe_plot_df = data_df[data_df.algorithm.str.contains('dedupe')].copy()
    dedupe_plot_df['specification'] = dedupe_plot_df.dedupe_sample_size.apply(lambda x: f"Sample size: {x:,}")
    dedupe_plot_df = dedupe_plot_df.groupby(['budget', 'ss_train', 'specification'])[metrics].mean().reset_index()
    dedupe_plot_df['budget_map'] = dedupe_plot_df['budget'].map(budget_map)
    
    # CREATE THE PLOT
    
    for metric_i, metric in enumerate(metrics):
        
        if 'error_pair' in metric:
            display(Markdown('<font color="red">NOTE: pair level, not "corr"</font>'))
        
        if diff_markers:
            g = sns.lmplot(
                data=dedupe_plot_df,
                x='budget_map',
                y=metric,
                hue='specification',
                col='ss_train',
                legend=False,
                aspect=0.9,
                ci=False,
                fit_reg=False,
                hue_order=hue_order,
                markers=[marker_map[m] for m in hue_order],
                scatter_kws={'s':150},
                palette=['#800000', '#FFB547', '#725663', '#5B8FA8']
            )
        else:
            g = sns.catplot(
                data=dedupe_plot_df,
                x='budget',
                y=metric,
                hue='specification',
                col='ss_train',
                legend=False,
                kind='point',
                ci=False,
                s=6,
                aspect=.8,
                scale=.7,
                hue_order=hue_order,
                palette=['#800000', '#FFB547', '#725663', '#5B8FA8']
            )
        _ = g.set_titles(col_template='N Admin Rows = {col_name}')
        
        if 'tpr' in metric:
            metric_str = 'Recall'
        elif 'prc' in metric: 
            metric_str = 'Precision'
        elif 'error' in metric:
            metric_str = 'Total Error'
        else: 
            metric_str = metric.capitalize()
            
        if metric_str == 'Precision':
            _ = plt.ylim(.9, 1.01)
            
        _ = plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            
        _ = g.set_ylabels(metric_str)
        _ = g.set_xlabels('Budget')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if metric_i == 1:
                _ = g.add_legend(title='', fontsize='large')
            else:
                # hack so the legend only print on the middle of three plot "rows"
                _ = g.add_legend(handles=[plt.plot([], ls="-", color='white')[0]], 
                                 labels=['Sample Size: 300,0000'],
                                 frameon=False, labelcolor='white')
        
        if diff_markers:
            for ax_row in g.axes:
                for ax in ax_row:
                    _ = ax.set_xticks(ticks=np.arange(1, 7, 1))
                    _ = g.set_xticklabels(labels=[20, 40, 80, 200, 500, 1000])
                    _ = ax.grid(False, axis='x')
        
        _ = g.fig.subplots_adjust(top=.8, right=.92) 
        _ = g.fig.suptitle(f'({alphabet_list[metric_i]})', fontsize='x-large')        
        
        plt.show()
        
        if savefig:
            filepath = os.path.join(figure_output_dir, f'sample_size_result__{sample_desc}__{metric}.png')
            g.savefig(filepath)

