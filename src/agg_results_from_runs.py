"""
Post-processing script.
Learning curve data are generated for different splits of a dataset. 
This script aggregates LC results from different runs (i.e., different data splits).

Example:
python src/agg_results_from_runs.py --res_dir trn/ml.ADRP_pocket1_dock.lc/
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import lightgbm as lgb

# File path
filepath = Path(__file__).resolve().parent

# Utils
from learningcurve.lrn_crv import LearningCurve
from learningcurve.lrn_crv_plot import plot_lc_agg
    
        
def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate learning curves.')

    # Input data
    parser.add_argument('--res_dir', required=True, default=None, type=str,
                        help='Global dir where learning curve are located (default: None).')
    args, other_args = parser.parse_known_args(args)
    return args


def agg_scores(run_dirs, prfx='run'):
    """ Aggregate results from LC runs. """
    scores = []
    
    for i, r in enumerate(run_dirs):
        dpath = Path(r)/'lrn_crv_scores.csv'
        if not dpath.exists():
            continue

        scr = pd.read_csv( dpath )
        scr.rename(columns={'fold0': prfx+str(i)}, inplace=True)
        if len(scores)==0:
            scores = scr
        else:
            scores = scores.merge(scr, on=['metric', 'tr_size', 'set'])

    run_col_names = [c for c in scores.columns if prfx in c]

    scores_mean   = scores[run_col_names].mean(axis=1)
    scores_median = scores[run_col_names].median(axis=1)
    scores_std    = scores[run_col_names].std(axis=1)
    # scores_iqr    = iqr(scores.iloc[:, 6:].values, axis=0)

    scores.insert(loc=3, column='mean',   value=scores_mean)
    scores.insert(loc=3, column='median', value=scores_median)
    scores.insert(loc=3, column='std',    value=scores_std)
    return scores


def run(args):
    res_dir = Path( args['res_dir'] ).resolve()
    trg_name = res_dir.name.split('.')[1]
    
    run_dirs = glob( str(res_dir/'run*') )
    prfx = 'run'
    scores = agg_scores( run_dirs, prfx=prfx )
    
    print('Training set sizes:', np.unique(scores.tr_size))
    
    te_scores = scores[ scores['set']=='te' ].reset_index(drop=True)
    te_scores_mae = scores[ (scores['metric']=='mean_absolute_error') & (scores['set']=='te') ].reset_index(drop=True)

    save = True
    if save:
        scores.to_csv(res_dir/'all_scores.csv', index=False)
        te_scores.to_csv(res_dir/'te_scores.csv', index=False)
        te_scores_mae.to_csv(res_dir/'te_scores_mae.csv', index=False)

    metrics = ['mean_absolute_error', 'r2']
    for metric_name in metrics:
        # metric_name = 'mean_absolute_error'
        tr_set = 'te'
    
        plot_args = {'xtick_scale': 'log2', 'ytick_scale': 'log2'}
        ax = plot_lc_agg(scores, metric_name=metric_name, prfx=prfx, tr_set='te', **plot_args)

        ax.set_title( trg_name )
        ax.legend(frameon=True, fontsize=10, loc='best')
        ax.grid(True)
        plt.tight_layout()

        # Save
        plt.savefig(res_dir/(f'lc.{trg_name}.{metric_name}.png'), dpi=200)
    

def main(args):
    args = parse_args(args)
    args = vars(args)
    score = run(args)
    return score
    

if __name__ == '__main__':
    main(sys.argv[1:])
