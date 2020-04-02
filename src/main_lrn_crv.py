""" 
This script generates learning curves.
You need to specify:
1. Function that creates the ML model.
   For example:
   
    def my_lgbm_classifier(**kwargs):
        model = lightgbm.LGBMClassifier(**kwargs)
        return model
    
    def my_dnn(input_dim, dr_rate=0.2, lr=0.001, initializer='he_uniform', batchnorm=False):
        layers = [1000, 500, 250]
        inputs = Input(shape=(input_dim,), name='inputs')
        x = Dense(layers[0], activation='relu', kernel_initializer=initializer)(inputs)
        x = Dense(layers[1], activation='relu', kernel_initializer=initializer)(x)
        x = Dense(layers[2], activation='relu', kernel_initializer=initializer)(x)
        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        opt = SGD(lr=lr, momentum=0.9)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
        return model

2. Initlization (input) parameters that initialize the model.
   For example:
    ml_init_args = {'input_dim': 1200, 'dr_rate': 0.35}
    
3. Fitting parameters for the model
   For example:
    ml_fit_args = {'input_dim': 1200, 'dr_rate': 0.35}
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time
from pprint import pprint, pformat
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
from utils.classlogger import Logger
from utils.utils import load_data, dump_dict, get_print_func
from ml.scale import scale_fea
from ml.data import extract_subset_fea
from learningcurve.lrn_crv import LearningCurve
import learningcurve.lrn_crv_plot as lrn_crv_plot 
    
        
def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate learning curves.')

    # Input data
    parser.add_argument('-dp', '--datapath', required=True, default=None, type=str,
                        help='Full path to data (default: None).')

    # Data splits
    parser.add_argument('-sd', '--splitdir', required=True, default=None, type=str,
                        help='Full path to data splits (default: None).')
    parser.add_argument('--split_id', default=0, type=int, help='Split id (default: 0).')
    
    # Outdir
    parser.add_argument('--gout', default=None, type=str,
                        help='Gloabl outdir. Path that will contain the runs (default: None).')
    parser.add_argument('--rout', default=None, type=str,
                        help='Run outdir. This is the path for the specific run (default: None).')

    # Select target to predict
    parser.add_argument('-t', '--trg_name', default='reg', type=str, choices=['reg', 'cls'],
                        help='Name of target variable (default: reg).')

    # Select feature types
    parser.add_argument('-df', '--drug_fea', nargs='+', default=['mod'], choices=['mod'],
                        help='Drug features (default: mod).')

    # Data scaling
    parser.add_argument('-sc', '--scaler', default=None, type=str, choices=['stnd', 'minmax', 'rbst'],
                        help='Feature normalization method (stnd, minmax, rbst) (default: None).')    
    
    # Data split methods
    # parser.add_argument('-cvf', '--cv_folds', default=1, type=str, help='Number cross-val folds (default: 1).')
    # parser.add_argument('-cvf_arr', '--cv_folds_arr', nargs='+', type=int, default=None, help='The specific folds in the cross-val run (default: None).')
    
    # ML models
    ## parser.add_argument('-ml', '--model_name', default='lgb_reg', type=str,
    ##                     choices=['lgb_reg', 'rf_reg', 'nn_reg', 'nn_reg0', 'nn_reg1', 'nn_reg_attn', 'nn_reg_layer_less', 'nn_reg_layer_more',
    ##                              'nn_reg_neuron_less', 'nn_reg_neuron_more', 'nn_reg_res', 'nn_reg_mini', 'nn_reg_ap'], help='ML model (default: lgb_reg).')

    # LightGBM params
    parser.add_argument('--gbm_leaves', default=31, type=int, help='Maximum tree leaves for base learners (default: 31).')
    parser.add_argument('--gbm_lr', default=0.1, type=float, help='Boosting learning rate (default: 0.1).')
    parser.add_argument('--gbm_max_depth', default=-1, type=int, help='Maximum tree depth for base learners (default: -1).')
    parser.add_argument('--gbm_trees', default=100, type=int, help='Number of trees (default: 100).')

    # Learning curve
    parser.add_argument('--lc_step_scale', default='log', type=str, choices=['log', 'linear'],
                        help='Scale of progressive sampling of shards in a learning curve (log2, log, log10, linear) (default: log).')
    parser.add_argument('--min_shard', default=128, type=int, help='The lower bound for the shard sizes (default: 128).')
    parser.add_argument('--max_shard', default=None, type=int, help='The upper bound for the shard sizes (default: None).')
    parser.add_argument('--n_shards', default=5, type=int, help='Number of shards (default: 5).')
    parser.add_argument('--shards_arr', nargs='+', type=int, default=None, help='List of the actual shards in the learning curve plot (default: None).')
    parser.add_argument('--save_model', action='store_true', help='Whether to trained models (default: False).')
    parser.add_argument('--plot_fit', action='store_true', help='Whether to generate the fit (default: False).')
    
    # HPs file
    parser.add_argument('--hp_file', default=None, type=str, help='File containing hyperparameters for training (default: None).')
    
    # HPO metric
    parser.add_argument('--hpo_metric', default='mean_absolute_error', type=str, choices=['mean_absolute_error'],
                        help='Metric for HPO evaluation. Required for UPF workflow on Theta HPC (default: mean_absolute_error).')    
    
    # Other
    parser.add_argument('--n_jobs', default=8, type=int, help='Default: 8.')
    parser.add_argument('--seed', default=0, type=int, help='Default: 0.')
    # parser.add_argument('--verbose', default=True, type=int, help='Default: True.')

    args, other_args = parser.parse_known_args(args)
    return args

    
# def get_model_kwargs(args):
#     """ Get ML model init and fit agrs. """
#     if args['framework'] == 'lightgbm':
#         model_init_kwargs = { 'n_estimators': args['gbm_trees'], 'max_depth': args['gbm_max_depth'],
#                               'learning_rate': args['gbm_lr'], 'num_leaves': args['gbm_leaves'],
#                               'n_jobs': args['n_jobs'], 'random_state': args['seed'] }
#         model_fit_kwargs = {'verbose': False}
#     elif args['framework'] == 'sklearn':
#         model_init_kwargs = { 'n_estimators': args['rf_trees'], 'n_jobs': args['n_jobs'], 'random_state': args['seed'] }
#         model_fit_kwargs = {}
#     elif args['framework'] == 'keras':
#         model_init_kwargs = { 'input_dim': data.shape[1], 'dr_rate': args['dr_rate'],
#                               'opt_name': args['opt'], 'lr': args['lr'], 'batchnorm': args['batchnorm']}
#         model_fit_kwargs = { 'batch_size': args['batch_size'], 'epochs': args['epochs'], 'verbose': 1 }        
#     return model_init_kwargs, model_fit_kwargs               


def run(args):
    t0 = time()
    datapath = Path( args['datapath'] ).resolve()
    splitdir = Path( args['splitdir'] ).resolve()
    split_id = args['split_id']

    # ML type ('reg' or 'cls')
    """
    if 'reg' in args['model_name']:
        mltype = 'reg'
    elif 'cls' in args['model_name']:
        mltype = 'cls'
    else:
        raise ValueError("model_name must contain 'reg' or 'cls'.")
    """
    
    # -----------------------------------------------
    #       Global outdir
    # -----------------------------------------------
    if args['gout'] is not None:
        gout = Path( args['gout'] )
    else:
        gout = filepath.parent / 'trn'
        gout = gout / datapath.with_suffix('.lc').name
        args['gout'] = str(gout)
    os.makedirs(gout, exist_ok=True)
    
    # -----------------------------------------------
    #       Run (single split) outdir
    # -----------------------------------------------
    # Create run outdir
    if args['rout'] is not None:
        rout = gout / args['rout']
    else:
        rout = gout / f'run_{split_id}'
        args['rout'] = rout  # str(rout)
    os.makedirs(rout, exist_ok=True)
    
    # -----------------------------------------------
    #       Logger
    # -----------------------------------------------
    ## lg = Logger(gout/'lc.log')
    lg = Logger(rout/'lc.log')
    print_fn = get_print_func(lg.logger)
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(args)}')
    dump_dict(args, outpath=rout/'trn.args.txt') # dump args.
    
    # -----------------------------------------------
    #       Load data
    # -----------------------------------------------
    print_fn('\nLoad master dataset.')
    data = load_data( datapath )
    print_fn('data.shape {}'.format(data.shape))
    
    # Get features (x), target (y), and meta
    fea_list = args['drug_fea']
    xdata = extract_subset_fea(data, fea_list=fea_list, fea_sep='.')
    meta = data.drop( columns=xdata.columns )
    ydata = meta[[ args['trg_name'] ]]
    del data
    
    # -----------------------------------------------
    #       Scale features
    # -----------------------------------------------
    xdata = scale_fea(xdata=xdata, scaler_name=args['scaler'])    
    
    # -----------------------------------------------
    #       Data splits
    # -----------------------------------------------
    split_pattern = f'1fold_s{split_id}_*_id.csv'
    single_split_files = glob( str(splitdir/split_pattern) )

    # Get indices for the split
    assert len(single_split_files) >= 2, f'The split {s} contains only one file.'
    for id_file in single_split_files:
        if 'tr_id' in id_file:
            tr_id = load_data( id_file )
        elif 'vl_id' in id_file:
            vl_id = load_data( id_file )
        elif 'te_id' in id_file:
            te_id = load_data( id_file )    
        
    # -----------------------------------------------
    #      ML model configs
    # -----------------------------------------------
    # CLR settings
    ## clr_keras_kwargs = {'mode': args['clr_mode'], 'base_lr': args['clr_base_lr'],
    ##                     'max_lr': args['clr_max_lr'], 'gamma': args['clr_gamma']}

    # Define ML model
    ## if 'lgb' in args['model_name']:
    ##     args['framework'] = 'lightgbm'
    ## elif args['model_name'] == 'rf_reg':
    ##     args['framework'] = 'sklearn'
    ## elif 'nn_' in args['model_name']:
    ##     args['framework'] = 'keras'        

    # ML model definitions
    args['framework'] = 'lightgbm'
    ml_model_def = lgb.LGBMRegressor
    mltype = 'reg'
    ml_init_args = { 'n_estimators': args['gbm_trees'], 'max_depth': args['gbm_max_depth'],
                     'learning_rate': args['gbm_lr'], 'num_leaves': args['gbm_leaves'],
                     'n_jobs': args['n_jobs'], 'random_state': args['seed'] }
    ml_fit_args = {'verbose': False, 'early_stopping_rounds': 10}
        
    # -----------------------------------------------
    #      Learning curve 
    # -----------------------------------------------        
    # LC args
    lrn_crv_init_kwargs = { 'cv': None, 'cv_lists': (tr_id, vl_id, te_id), ## 'cv_folds_arr': args['cv_folds_arr'],
                            'lc_step_scale': args['lc_step_scale'], 'n_shards': args['n_shards'],
                            'min_shard': args['min_shard'], 'max_shard': args['max_shard'], 'outdir': args['rout'],
                            'shards_arr': args['shards_arr'], ## 'args': args,
                            'logger': lg.logger}
                    
    lrn_crv_trn_kwargs = { 'framework': args['framework'], 'mltype': mltype, 'ml_model_def': ml_model_def,
                           'n_jobs': args['n_jobs'], 'random_state': args['seed'],
                           ## 'model_name': args['model_name'],
                           'ml_model_def': ml_model_def}
    
    # LC object
    lc = LearningCurve( X=xdata, Y=ydata, meta=meta, **lrn_crv_init_kwargs )        

    if args['hp_file'] is None:
        # The regular workflow where all subsets are trained with the same HPs
        # model_init_kwargs, model_fit_kwargs = get_model_kwargs( args )
        # lrn_crv_trn_kwargs['init_kwargs'] = model_init_kwargs
        # lrn_crv_trn_kwargs['fit_kwargs'] = model_fit_kwargs

        lrn_crv_trn_kwargs['init_kwargs'] = ml_init_args
        lrn_crv_trn_kwargs['fit_kwargs'] = ml_fit_args
        
        lrn_crv_scores = lc.trn_learning_curve( **lrn_crv_trn_kwargs )
    else:
        # The workflow follows PS-HPO where we a the set HPs per subset.
        # In this case we need to call the trn_learning_curve() method for
        # every subset with appropriate HPs. We'll need to update shards_arr
        # for every run of trn_learning_curve().
        fpath = verify_dirpath(args['hp_file'])
        hp = pd.read_csv(fpath)
        hp.to_csv(args['rout']/'hpo_ps.csv', index=False)

        # Params to update based on framework
        if args['framework'] == 'lightgbm':
            prm_names = ['gbm_trees', 'gbm_max_depth', 'gbm_lr', 'gbm_leaves']
        elif args['framework'] == 'sklearn':
            prm_names = ['rf_trees']
        elif args['framework'] == 'keras':
            prm_names = ['dr_rate', 'opt', 'lr', 'batchnorm', 'batch_size']

        # Params of interest
        df_print = hp[ prm_names + ['tr_size', 'mean_absolute_error'] ]
        lg.logger.info( df_print )

        # Find the intersect btw available and requested tr sizes
        tr_sizes = list( set(lc.tr_shards).intersection(set(hp['tr_size'].unique())) )
        lg.logger.info('\nIntersect btw available and requested tr sizes: {}'.format( tr_sizes ))

        lrn_crv_scores = []
        for sz in tr_sizes:
            prm = hp[hp['tr_size']==sz]
            lrn_crv_init_kwargs['shards_arr'] = [sz]
            lc.tr_shards = [sz] 
            
            # Update model_init and model_fit params
            prm = prm.to_dict(orient='records')[0]  # unroll single-raw df into dict
                
            # Update args
            lg.logger.info('\nUpdate args for tr size {}'.format(sz))
            lg.logger.info( df_print[ df_print['tr_size']==sz ] )
            for n in prm_names:
                lg.logger.info('{}: set to {}'.format(n, prm[n]))
                args[n] = prm[n]

            model_init_kwargs, model_fit_kwargs = get_model_kwargs(args)
            lrn_crv_trn_kwargs['init_kwargs'] = model_init_kwargs
            lrn_crv_trn_kwargs['fit_kwargs'] = model_fit_kwargs

            per_subset_scores = lc.trn_learning_curve( **lrn_crv_trn_kwargs )
            lrn_crv_scores.append( per_subset_scores )

        # Concat per-subset scores 
        lrn_crv_scores = pd.concat(lrn_crv_scores, axis=0)

        # Save tr, vl, te separently
        lrn_crv_scores[ lrn_crv_scores['set']=='tr' ].to_csv( args['rout']/'tr_lrn_crv_scores.csv', index=False) 
        lrn_crv_scores[ lrn_crv_scores['set']=='vl' ].to_csv( args['rout']/'vl_lrn_crv_scores.csv', index=False) 
        lrn_crv_scores[ lrn_crv_scores['set']=='te' ].to_csv( args['rout']/'te_lrn_crv_scores.csv', index=False) 

    # Dump all scores
    lrn_crv_scores.to_csv( args['rout']/'lrn_crv_scores.csv', index=False)

    # Load results and plot
    lrn_crv_plot.plot_lrn_crv_all_metrics( lrn_crv_scores, outdir=args['rout'] )
    lrn_crv_plot.plot_lrn_crv_all_metrics( lrn_crv_scores, outdir=args['rout'], xtick_scale='log2', ytick_scale='log2' )
    
    # Dump args
    dump_dict(args, outpath=args['rout']/'args.txt')
    
    
    # ====================================
    if args['plot_fit']:
        figsize = (7, 5.5)
        metric_name = 'mean_absolute_error'
        xtick_scale, ytick_scale = 'log2', 'log2'
        plot_args = {'metric_name': metric_name, 'xtick_scale': xtick_scale, 'ytick_scale': xtick_scale, 'figsize': figsize}

        scores_te = lrn_crv_scores[(lrn_crv_scores.metric==metric_name) & (lrn_crv_scores.set=='te')].sort_values('tr_size').reset_index(drop=True)

        # -----  Finally plot power-fit  -----
        tot_pnts = len(scores_te['tr_size'])
        n_pnts_fit = 8 # Number of points to use for curve fitting starting from the largest size

        y_col_name = 'fold0'
        ax = None

        ax = lrn_crv_plot.plot_lrn_crv_new(
                x=scores_te['tr_size'][0:], y=scores_te[y_col_name][0:],
                ax=ax, ls='', marker='v', alpha=0.7,
                **plot_args, label='Raw Data')

        shard_min_idx = 0 if tot_pnts < n_pnts_fit else tot_pnts - n_pnts_fit

        ax, _, gof = lrn_crv_plot.plot_lrn_crv_power_law(
                x=scores_te['tr_size'][shard_min_idx:], y=scores_te[y_col_name][shard_min_idx:],
                **plot_args, plot_raw=False, ax=ax, alpha=1 );

        ax.legend(frameon=True, fontsize=10, loc='best')
        plt.tight_layout()
        plt.savefig(args['outdir']/f'power_law_fit_{metric_name}.png')


        # -----  Extrapolation  -----
        n_pnts_ext = 1 # Number of points to extrapolate to
        n_pnts_fit = 6 # Number of points to use for curve fitting starting from the largest size

        tot_pnts = len(scores_te['tr_size'])
        m0 = tot_pnts - n_pnts_ext
        shard_min_idx = m0 - n_pnts_fit

        ax = None

        # Plot of all the points
        ax = lrn_crv_plot.plot_lrn_crv_new(
                x=scores_te['tr_size'][0:shard_min_idx], y=scores_te[y_col_name][0:shard_min_idx],
                ax=ax, ls='', marker='v', alpha=0.8, color='k',
                **plot_args, label='Excluded Points')

        # Plot of all the points
        ax = lrn_crv_plot.plot_lrn_crv_new(
                x=scores_te['tr_size'][shard_min_idx:m0], y=scores_te[y_col_name][shard_min_idx:m0],
                ax=ax, ls='', marker='*', alpha=0.8, color='g',
                **plot_args, label='Included Points')

        # Extrapolation
        ax, _, mae_et = lrn_crv_plot.lrn_crv_power_law_extrapolate(
                x=scores_te['tr_size'][shard_min_idx:], y=scores_te[y_col_name][shard_min_idx:],
                n_pnts_ext=n_pnts_ext,
                **plot_args, plot_raw_it=False, label_et='Extrapolation', ax=ax );

        ax.legend(frameon=True, fontsize=10, loc='best')
        plt.savefig(args['outdir']/f'power_law_ext_{metric_name}.png')

    # ====================================
    if (time()-t0)//3600 > 0:
        print_fn('Runtime: {:.1f} hrs'.format( (time()-t0)/3600) )
    else:
        print_fn('Runtime: {:.1f} min'.format( (time()-t0)/60) )
    
    print_fn('Done.')
    lg.kill_logger()
    del xdata, ydata

    # This is required for HPO via UPF workflow on Theta HPC
    return lrn_crv_scores[(lrn_crv_scores['metric'] == args['hpo_metric']) & (lrn_crv_scores['set'] == 'te')].values[0][3]


def main(args):
    args = parse_args(args)
    args = vars(args)
    score = run(args)
    return score
    

if __name__ == '__main__':
    main(sys.argv[1:])


