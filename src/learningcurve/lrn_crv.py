"""
Functions to generate learning curves.
Records performance (error or score) vs training set size.
"""
import os
import sys
from pathlib import Path
from time import time

import sklearn
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from sklearn import metrics
from math import sqrt
from scipy import optimize

from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

from datasplit import splitter
from datasplit.splitter import data_splitter

# try:
#     import tensorflow as tf
#     # print(tf.__version__)
#     if int(tf.__version__.split('.')[0]) < 2:
#         # print('Load keras standalone package.')
#         import keras
#         from keras.models import load_model
#         from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
#         from keras.utils import plot_model
#     else:
#         # print('Load keras from tf.')
#         # from tensorflow import keras
#         from tensorflow.python.keras.models import load_model
#         from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
#         from tensorflow.python.keras.utils import plot_model
# except:
#     print('Could not import tensorflow.')

# import keras
# from keras.models import load_model
# from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
# from keras.utils import plot_model

# Utils
filepath = Path(__file__).resolve().parent
sys.path.append( os.path.abspath(filepath/'../ml') )
sys.path.append( os.path.abspath(filepath/'../utils') )
sys.path.append( os.path.abspath(filepath/'../datasplit') )

# import ml.ml_models as ml_models
from ml.keras_utils import save_krs_history, plot_prfrm_metrics, r2_krs
from ml.evals import calc_preds, calc_scores, dump_preds
from utils.utils import dump_dict
from utils.plots import plot_hist, plot_runtime


# --------------------------------------------------------------------------------
class LearningCurve():
    """
    Train estimator using multiple shards (train set sizes) and generate learning curves for multiple performance metrics.
    Example:
        lc = LearningCurve(xdata, ydata, cv_lists=(tr_ids, vl_ids))
        lc_scores = lc.trn_learning_curve(
            framework=framework, mltype=mltype, model_name=model_name,
            ml_init_args=ml_init_args, ml_fit_args=ml_fit_args, clr_keras_args=clr_keras_args)
    """
    def __init__(self,
            X, Y,
            meta=None,
            # cv=5,
            cv_lists=None,  # (tr_id, vl_id, te_id)
            cv_folds_arr=None,
            n_splits=1,
            lc_step_scale: str='log2',
            min_shard = 0,
            max_shard = None,
            n_shards: int=None,
            shards_arr: list=[],
            args=None,
            print_fn=print,
            save_model=False,
            outdir=Path('./')):
        """
        Args:
            X : array-like (pd.DataFrame or np.ndarray)
            Y : array-like (pd.DataFrame or np.ndarray)
            meta : array-like file of metadata (each item corresponds to an (x,y) sample
            cv : (optional) number of cv folds (int) or sklearn cv splitter --> scikit-learn.org/stable/glossary.html#term-cv-splitter
            cv_lists : tuple of 3 dicts, cv_lists[0] and cv_lists[1], cv_lists[2], that contain the tr, vl, and te folds, respectively
            cv_folds_arr : list that contains the specific folds in the cross-val run

            lc_step_scale : specifies how to generate the shard values. 
                Available values: 'linear', 'log2', 'log10'.

            min_shard : min shard value in the case when lc_step_scale is 'log2' or 'log10'
            max_shard : max shard value in the case when lc_step_scale is 'log2' or 'log10'

            n_shards : number of shards in the learning curve (used only in the lc_step_scale is 'linear')
            shards_arr : list of ints specifying the shards to process (e.g., [128, 256, 512])
            
            shard_frac : list of relative numbers of training samples that are used to generate learning curves
                e.g., shard_frac=[0.1, 0.2, 0.4, 0.7, 1.0].
                If this arg is not provided, then the training shards are generated from n_shards and lc_step_scale.
                
            args : command line args
            save_model : dump model if True (keras model ignores this arg since we load the best model to calc score)
        """
        self.X = pd.DataFrame(X).reset_index(drop=True)
        self.Y = pd.DataFrame(Y).reset_index(drop=True)
        self.meta = pd.DataFrame(meta).reset_index(drop=True)

        self.cv_lists = cv_lists
        self.cv_folds_arr = cv_folds_arr

        self.n_splits = n_splits

        self.lc_step_scale = lc_step_scale 
        self.min_shard = min_shard
        self.max_shard = max_shard
        self.n_shards = n_shards
        self.shards_arr = shards_arr

        ## self.args = args
        ## self.print_fn = get_print_func(logger)
        self.print_fn = print_fn
        self.save_model = save_model
        self.outdir = Path( outdir )

        self.create_fold_dcts()
        self.create_tr_shards_list()
        # self.trn_single_subset() # TODO: implement this method for better modularity

        
    def create_fold_dcts(self):
        """ Converts a tuple of arrays self.cv_lists into two dicts, tr_dct, vl_dct, and te_dict.
        Both sets of data structures contain the splits of all the k-folds. """
        tr_dct = {}
        vl_dct = {}
        te_dct = {}

        # Use lists passed as input arg
        if self.cv_lists is not None:
            tr_id = self.cv_lists[0]
            vl_id = self.cv_lists[1]
            te_id = self.cv_lists[2]
            assert (tr_id.shape[1]==vl_id.shape[1]) and (tr_id.shape[1]==te_id.shape[1]), 'tr, vl, and te must have the same number of folds.'
            self.cv_folds = tr_id.shape[1]

            # Calc the split ratio if cv=1
            if self.cv_folds == 1:
                total_samples = tr_id.shape[0] + vl_id.shape[0] + te_id.shape[0]
                self.vl_size = vl_id.shape[0] / total_samples
                self.te_size = te_id.shape[0] / total_samples

            if self.cv_folds_arr is None:
                self.cv_folds_arr = [f+1 for f in range(self.cv_folds)]
                
            for fold in range(tr_id.shape[1]):
                # cv_folds_arr contains the specific folds we wish to process
                if fold+1 in self.cv_folds_arr:
                    tr_dct[fold] = tr_id.iloc[:, fold].dropna().values.astype(int).tolist()
                    vl_dct[fold] = vl_id.iloc[:, fold].dropna().values.astype(int).tolist()
                    te_dct[fold] = te_id.iloc[:, fold].dropna().values.astype(int).tolist()


        # Generate folds on the fly if no pre-defined folds were passed
        # TODO: this option won't work after we added test set in addition to train and val sets.
        else:
            # raise ValueError('This option is not supported.')
            
            if isinstance(self.cv, int):
                # By default, it k-fold cross-validation
                self.cv_folds = self.cv
                self.cv = KFold(n_splits=self.cv_folds, shuffle=False, random_state=self.random_state)
            """
            else:
                # cv is sklearn splitter
                self.cv_folds = cv.get_n_splits() 

            if cv_folds == 1:
                self.vl_size = cv.test_size

            # Create sklearn splitter 
            if self.mltype == 'cls':
                if self.Y.ndim > 1 and self.Y.shape[1] > 1:
                    splitter = self.cv.split(self.X, np.argmax(self.Y, axis=1))
            else:
                splitter = self.cv.split(self.X, self.Y)
            
            # Generate the splits
            for fold, (tr_vec, vl_vec) in enumerate(splitter):
                tr_dct[fold] = tr_vec
                vl_dct[fold] = vl_vec
            """

        # Keep dicts
        self.tr_dct = tr_dct
        self.vl_dct = vl_dct
        self.te_dct = te_dct


    def create_tr_shards_list(self):
        """ Generate a list of training shards (training sizes). """
        if self.shards_arr is not None:
            # No need to generate an array of training shards if shards_arr is specified
            self.tr_shards = self.shards_arr
            
        else:
            # Fixed spacing
            if self.max_shard is None:
                key = list(self.tr_dct.keys())[0]
                self.max_shard = len(self.tr_dct[key]) # total number of available training samples

            # Full vector of shards
            # (we create a vector with very large values so that we later truncate it with max_shard)
            scale = self.lc_step_scale.lower()
            if scale == 'linear':
                m = np.linspace(self.min_shard, self.max_shard, self.n_shards+1)[1:]
            else:
                # we create very large vector m, so that we later truncate it with max_shard
                if scale == 'log2':
                    m = 2 ** np.array(np.arange(30))[1:]
                # elif scale == 'log':
                    # m = np.exp( np.array(np.arange(8))[1:] )
                elif scale == 'log10':
                    m = 10 ** np.array(np.arange(8))[1:]
                elif scale == 'log':
                    if self.n_shards is not None:
                        # TODO: need to update code to follow this methodology. This can
                        # allow to remove almost all the code at the bottom 
                        # self.min_shard = 0 if self.min_shard is None else self.min_shard
                        # www.researchgate.net/post/is_the_logarithmic_spaced_vector_the_same_in_any_base
                        pw = np.linspace(0, self.n_shards-1, num=self.n_shards) / (self.n_shards-1)
                        m = self.min_shard * (self.max_shard/self.min_shard) ** pw
                        # m = 2 ** np.linspace(self.min_shard, self.max_shard, self.n_shards)
                        m = np.array( [int(i) for i in m] )
                        self.tr_shards = m
                        self.print_fn('\nTrain shards: {}\n'.format(self.tr_shards))
                        return None
                        

            m = np.array( [int(i) for i in m] ) # cast to int

            # Set min shard
            idx_min = np.argmin( np.abs( m - self.min_shard ) )
            if m[idx_min] > self.min_shard:
                m = m[idx_min:]  # all values larger than min_shard
                m = np.concatenate( (np.array([self.min_shard]), m) )  # preceed arr with specified min_shard
            else:
                m = m[idx_min:]

            # Set max shard
            idx_max = np.argmin( np.abs( m - self.max_shard ) )
            if m[idx_max] > self.max_shard:
                m = list(m[:idx_max])    # all values EXcluding the last one
                m.append(self.max_shard)
            else:
                m = list(m[:idx_max+1])  # all values INcluding the last one
                m.append(self.max_shard) # TODO: should we append this??
                # If the diff btw max_samples and the latest shards (m[-1] - m[-2]) is "too small",
                # then remove max_samples from the possible shards.
                if 0.5*m[-3] > (m[-1] - m[-2]): m = m[:-1] # heuristic to drop the last shard

            self.tr_shards = m
        # --------------------------------------------
        
        self.print_fn('\nTrain shards: {}\n'.format(self.tr_shards))


    def trn_learning_curve(self,
            framework: str,  # 'lightgbm'
            mltype: str,         # 'reg'
            ## model_name: str, # 'lgb_reg'

            ml_model_def,
            keras_callbacks_def,

            ml_init_args: dict={},
            ml_fit_args: dict={},
            ## clr_keras_args: dict={},
            metrics: list=['r2', 'neg_mean_absolute_error', 'neg_median_absolute_error', 'neg_mean_squared_error'],
            n_jobs: int=4,
            random_state: int=None,
            plot=True):
        """ 
        Args:
            framework : ml framework (keras, lightgbm, or sklearn)
            mltype : type to ml problem (reg or cls)
            ml_init_args : dict of parameters that initializes the estimator
            ml_fit_args : dict of parameters to the estimator's fit() method
            clr_keras_args : 
            metrics : allow to pass a string of metrics  TODO!
        """
        self.framework = framework
        self.mltype = mltype
        ## self.model_name = model_name
        
        self.ml_model_def = ml_model_def
        self.keras_callbacks_def = keras_callbacks_def
        
        self.ml_init_args = ml_init_args
        self.ml_fit_args = ml_fit_args
        ## self.clr_keras_args = clr_keras_args
        self.metrics = metrics
        self.n_jobs = n_jobs
        # self.random_state = random_state
        
        # Start nested loop of train size and cv folds
        tr_scores_all = [] # list of dicts
        vl_scores_all = [] # list of dicts
        te_scores_all = [] # list of dicts

        # Record runtime per shard
        runtime_records = []

        # CV loop
        # TODO: consider removing this since we don't use CV loops anymore!
        for fold_num in self.tr_dct.keys():
            # fold = fold + 1
            self.print_fn(f'Fold {fold_num} out of {list(self.tr_dct.keys())}')    

            # Get the indices for this fold
            tr_id = self.tr_dct[fold_num]
            vl_id = self.vl_dct[fold_num]
            te_id = self.te_dct[fold_num]
            
            # Extract Train set T, Validation set V, and Test set E
            xtr, ytr, mtr = self.get_data_by_id(tr_id) # samples from xtr are sequentially sampled for TRAIN
            xvl, yvl, mvl = self.get_data_by_id(vl_id) # fixed set of VAL samples for the current CV split
            xte, yte, mte = self.get_data_by_id(te_id) # fixed set of TEST samples for the current CV split

            xvl = np.asarray(xvl)
            yvl = np.asarray(yvl)
            xte = np.asarray(xte)
            yte = np.asarray(yte)            
            
            # Shards loop (iterate across the dataset sizes and train)
            """
            np.random.seed(random_state)
            idx = np.random.permutation(len(xtr))
            Note that we don't shuffle the dataset another time using the commands above.
            """
            # idx = np.arange(len(xtr))
            for i, tr_sz in enumerate(self.tr_shards):
                # For each shard: train model, save best model, calc tr_scores, calc_vl_scores
                self.print_fn(f'\tTrain size: {tr_sz} ({i+1}/{len(self.tr_shards)})')   

                # Sequentially get a subset of samples (the input dataset X must be shuffled)
                # TODO: why not to use simply xtr.iloc[:tr_sz, :]
                # xtr_sub = xtr.loc[idx[:tr_sz], :]
                # ytr_sub = ytr.loc[idx[:tr_sz]]  # np.squeeze(ytr[idx[:tr_sz], :])
                # mtr_sub = mtr.loc[idx[:tr_sz], :]
                xtr_sub = xtr.iloc[:tr_sz, :]
                ytr_sub = ytr.iloc[:tr_sz]
                mtr_sub = mtr.iloc[:tr_sz, :]
                
                xtr_sub = np.asarray(xtr_sub)
                ytr_sub = np.asarray(ytr_sub)                
                
                # Get the estimator
                model = self.ml_model_def( **self.ml_init_args )
                
                # Train
                # TODO: consider to pass and function train_model that will be used to train model and return
                # specified parameters, or a dict with required and optional parameters
                eval_set = (xvl, yvl)
                if self.framework=='lightgbm':
                    model, trn_outdir, runtime = self.trn_lgbm_model(model=model, xtr_sub=xtr_sub, ytr_sub=ytr_sub,
                                                                     fold=fold_num, tr_sz=tr_sz, eval_set=eval_set)
                elif self.framework=='sklearn':
                    model, trn_outdir, runtime = self.trn_sklearn_model(model=model, xtr_sub=xtr_sub, ytr_sub=ytr_sub,
                                                                        fold=fold_num, tr_sz=tr_sz, eval_set=None)
                elif self.framework=='keras':
                    model, trn_outdir, runtime = self.trn_keras_model(model=model, xtr_sub=xtr_sub, ytr_sub=ytr_sub,
                                                                      fold=fold_num, tr_sz=tr_sz, eval_set=eval_set)
                elif self.framework=='pytorch':
                    pass
                else:
                    raise ValueError(f'Framework {self.framework} is not yet supported.')
                    
                if model is None:
                    continue # sometimes keras fails to train a model (evaluates to nan)

                # Dump args
                model_args = self.ml_init_args.copy()
                model_args.update( self.ml_fit_args )
                dump_dict(model_args, trn_outdir/'model_args.txt') 

                # Save plot of target distribution
                plot_hist(ytr_sub, title=f'(Train size={tr_sz})',   path=trn_outdir/'hist_tr.png')
                plot_hist(yvl,     title=f'(Val size={len(yvl)})',  path=trn_outdir/'hist_vl.png')
                plot_hist(yte,     title=f'(Test size={len(yte)})', path=trn_outdir/'hist_te.png')
                    
                # Calc preds and scores
                # ... training set
                y_pred, y_true = calc_preds(model, x=xtr_sub, y=ytr_sub, mltype=self.mltype)
                tr_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=self.mltype, metrics=None)
                dump_preds(y_true, y_pred, meta=mtr_sub, outpath=trn_outdir/'preds_tr.csv')
                # ... val set
                y_pred, y_true = calc_preds(model, x=xvl, y=yvl, mltype=self.mltype)
                vl_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=self.mltype, metrics=None)
                dump_preds(y_true, y_pred, meta=mvl, outpath=trn_outdir/'preds_vl.csv')
                # ... test set
                y_pred, y_true = calc_preds(model, x=xte, y=yte, mltype=self.mltype)
                te_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=self.mltype, metrics=None)
                dump_preds(y_true, y_pred, meta=mte, outpath=trn_outdir/'preds_te.csv')
                
                # del estimator, model
                del model

                # Store runtime
                runtime_records.append((fold_num, tr_sz, runtime))

                # Add metadata
                tr_scores['set'] = 'tr'
                tr_scores['fold'] = 'fold'+str(fold_num)
                tr_scores['tr_size'] = tr_sz
                
                vl_scores['set'] = 'vl'
                vl_scores['fold'] = 'fold'+str(fold_num)
                vl_scores['tr_size'] = tr_sz

                te_scores['set'] = 'te'
                te_scores['fold'] = 'fold'+str(fold_num)
                te_scores['tr_size'] = tr_sz

                # Append scores (dicts)
                tr_scores_all.append(tr_scores)
                vl_scores_all.append(vl_scores)
                te_scores_all.append(te_scores)

                # Dump intermediate scores
                scores = pd.concat([scores_to_df([tr_scores]), scores_to_df([vl_scores]), scores_to_df([te_scores])], axis=0)
                scores.to_csv( trn_outdir/'scores.csv', index=False )
                del trn_outdir, scores
                
            # Dump intermediate results (this is useful if the run terminates before run ends)
            scores_all_df_tmp = pd.concat([scores_to_df(tr_scores_all), scores_to_df(vl_scores_all), scores_to_df(te_scores_all)], axis=0)
            scores_all_df_tmp.to_csv( self.outdir / ('tmp_lc_scores_cv' + str(fold_num) + '.csv'), index=False )

        # Scores to df
        tr_scores_df = scores_to_df( tr_scores_all )
        vl_scores_df = scores_to_df( vl_scores_all )
        te_scores_df = scores_to_df( te_scores_all )
        scores_df = pd.concat([tr_scores_df, vl_scores_df, te_scores_df], axis=0)
        
        # Dump final results
        tr_scores_df.to_csv( self.outdir/'tr_lc_scores.csv', index=False) 
        vl_scores_df.to_csv( self.outdir/'vl_lc_scores.csv', index=False) 
        te_scores_df.to_csv( self.outdir/'te_lc_scores.csv', index=False) 
        scores_df.to_csv( self.outdir/'lc_scores.csv', index=False) 

        # Runtime df
        runtime_df = pd.DataFrame.from_records(runtime_records, columns=['fold', 'tr_sz', 'time'])
        runtime_df.to_csv( self.outdir/'runtime.csv', index=False) 

        return scores_df
    
    
    def get_data_by_id(self, idx):
        """ Returns a tuple of (features (x), target (y), metadata (m))
        for an input array of indices (idx). """
        # x_data = self.X[idx, :]
        # y_data = np.squeeze(self.Y[idx, :])        
        # m_data = self.meta.loc[idx, :]
        # x_data = self.X.loc[idx, :].reset_index(drop=True)
        # y_data = np.squeeze(self.Y.loc[idx, :]).reset_index(drop=True)
        # m_data = self.meta.loc[idx, :].reset_index(drop=True)
        # return x_data, y_data, m_data
        x_data = self.X.iloc[idx, :].reset_index(drop=True)
        y_data = np.squeeze(self.Y.iloc[idx, :]).reset_index(drop=True)
        if self.meta is not None:
            m_data = self.meta.iloc[idx, :].reset_index(drop=True)
        else:
            meta = None
        return x_data, y_data, m_data    


    def trn_keras_model(self, model, xtr_sub, ytr_sub, fold, tr_sz, eval_set=None):
        """ Train and save Keras model. """
        trn_outdir = self.create_trn_outdir(fold, tr_sz)
        # keras.utils.plot_model(model, to_file=self.outdir/'nn_model.png') # comment this when using Theta
                
        # if bool(self.clr_keras_args):
        ## if self.clr_keras_args['mode'] is not None:
        ##     keras_callbacks.append( ml_models.clr_keras_callback(**self.clr_keras_args) )

        # Fit params
        ml_fit_args = self.ml_fit_args.copy()
        ml_fit_args['validation_data'] = eval_set
        ml_fit_args['callbacks'] = self.keras_callbacks_def( trn_outdir )
        
        # Train model
        t0 = time()
        history = model.fit(xtr_sub, ytr_sub, **ml_fit_args)
        runtime = (time() - t0)/60
        save_krs_history(history, outdir=trn_outdir)
        plot_prfrm_metrics(history, title=f'Train size: {tr_sz}', skp_ep=10, add_lr=True, outdir=trn_outdir)

        # Remove key (we'll dump this dict so we don't need to print all the eval set)
        # ml_fit_args.pop('validation_data', None)
        # ml_fit_args.pop('callbacks', None)

        # Load the best model (https://github.com/keras-team/keras/issues/5916)
        # model = keras.models.load_model(str(trn_outdir/'model_best.h5'), custom_objects={'r2_krs': ml_models.r2_krs})
        model_path = trn_outdir / 'model_best.h5'
        if model_path.exists():
            # model = keras.models.load_model( str(model_path) )
            import tensorflow as tf
            model = tf.keras.models.load_model( str(model_path),
                                                custom_objects={'r2_krs': r2_krs} )
        else:
            model = None
        return model, trn_outdir, runtime


    def trn_lgbm_model(self, model, xtr_sub, ytr_sub, fold, tr_sz, eval_set=None):
        """ Train and save LigthGBM model. """
        trn_outdir = self.create_trn_outdir(fold, tr_sz)
        
        # Fit params
        ml_fit_args = self.ml_fit_args.copy()
        ml_fit_args['eval_set'] = eval_set  
        
        # Train and save model
        t0 = time()
        model.fit(xtr_sub, ytr_sub, **ml_fit_args)
        runtime = (time() - t0)/60

        # Remove key (we'll dump this dict so we don't need to print all the eval set)
        ml_fit_args.pop('eval_set', None)

        if self.save_model:
            joblib.dump(model, filename = trn_outdir / ('model.'+self.model_name+'.pkl') )
        return model, trn_outdir, runtime
    
    
    def trn_sklearn_model(self, model, xtr_sub, ytr_sub, fold, tr_sz, eval_set=None):
        """ Train and save sklearn model. """
        trn_outdir = self.create_trn_outdir(fold, tr_sz)
        
        # Fit params
        ml_fit_args = self.ml_fit_args
        # ml_fit_args = self.ml_fit_args.copy()

        # Train and save model
        t0 = time()
        model.fit(xtr_sub, ytr_sub, **ml_fit_args)
        runtime = (time() - t0)/60
        if self.save_model:
            joblib.dump(model, filename = trn_outdir / ('model.'+self.model_name+'.pkl') )
        return model, trn_outdir, runtime
    
    
    def create_trn_outdir(self, fold, tr_sz):
        trn_outdir = self.outdir / ('cv'+str(fold) + '_sz'+str(tr_sz))
        os.makedirs(trn_outdir, exist_ok=True)
        return trn_outdir
# --------------------------------------------------------------------------------


def scores_to_df(scores_all):
    """ (tricky commands) """
    df = pd.DataFrame(scores_all)
    df = df.melt(id_vars=['fold', 'tr_size', 'set'])
    df = df.rename(columns={'variable': 'metric'})
    df = df.pivot_table(index=['metric', 'tr_size', 'set'], columns=['fold'], values='value')
    df = df.reset_index(drop=False)
    df.columns.name = None
    return df


