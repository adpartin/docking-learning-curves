"""
Post-processing script.
Learning curve data are generated for different splits of a dataset. 
This script aggregates LC results from different runs (i.e., different data splits).

Example:
python src/agg_results_from_runs.py --res_dir trn/ml.ADRP_pocket1_dock.lc/
python src/agg_results_from_runs.py --res_dir trn/ml.ADRP-ADPR_pocket1_dock.lc/
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

# File path
filepath = Path(__file__).resolve().parent

# Utils
from learningcurve.lc_plots import plot_lc_many_metric # plot_lc_agg
    
        
def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate learning curves.')

    # Input data
    parser.add_argument('--res_dir', required=True, default=None, type=str,
                        help='Global dir where learning curve are located (default: None).')
    args, other_args = parser.parse_known_args(args)
    return args


def agg_scores( run_dirs ):
    """ Aggregate results from LC runs. """
    scores = []
    
    for i, r in enumerate(run_dirs):
        dpath = Path(r)/'lc_scores.csv'
        if not dpath.exists():
            continue

        scr = pd.read_csv( dpath )
        scr['split'] = Path(r).name
        scores.append( scr )

    scores = pd.concat( scores, axis=0 )
    return scores


def run(args):
    res_dir = Path( args['res_dir'] ).resolve()
    dir_name = res_dir.name # .split('.')[1]
    
    run_dirs = glob( str(res_dir/'run*') )
    scores = agg_scores( run_dirs )
    
    print('Training set sizes:', np.unique(scores.tr_size))
    
    te_scores = scores[ scores['set']=='te' ].reset_index(drop=True)
    te_scores_mae = scores[ (scores['metric']=='mean_absolute_error') & (scores['set']=='te') ].reset_index(drop=True)

    save = True
    if save:
        scores.to_csv(res_dir/'all_scores.csv', index=False)
        te_scores.to_csv(res_dir/'te_scores.csv', index=False)
        te_scores_mae.to_csv(res_dir/'te_scores_mae.csv', index=False)

    kwargs = {'tr_set': 'te', 'xtick_scale': 'log2', 'ytick_scale': 'log2'}
    ax = plot_lc_many_metric(scores, metrics=['mean_absolute_error', 'r2'], outdir=res_dir, **kwargs)


def main(args):
    args = parse_args(args)
    args = vars(args)
    score = run(args)
    return score
    

if __name__ == '__main__':
    main(sys.argv[1:])
