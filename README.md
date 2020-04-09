The leaning curve methods currently support scikit-learn, LightGBM, and tf.keras ML models.

## Getting Started
#### Get the code
Clone the GitHub repo to get the necessary code. 
```
$ git clone https://github.com/adpartin/docking-learning-curves
```
#### Get the data
Create `./data` in the project dir and copy a batch of docking results of (ML dataframes). For example `./data/docking_data_march_30/ml.<*>.parquet`.<br>
To genearte ML docking dataframes, refer to `https://github.com/2019-ncovgroup/ML-docking-dataframe-generator`.

## Examples
1. Serial execution (generates train/val/test splits on the fly; `n_splits` serial runs). Call `src/main_lc.py`.
```
$ python src/main_lc.py --datapath data/docking_data_march_30/ml.ADRP-ADPR_pocket1_dock.parquet --n_splits 3 --lc_sizes 4 --max_size 185000 --gout ./trn_lc_main_serial_fly --trg_name reg
```

2. Serial execution (generates train/val/test splits on the fly; `n_splits` serial runs). Call `src/batch_lc.py` with `par_jobs` set to 1. This equivalent to example (1).
```
$ python src/batch_lc.py --datapath data/docking_data_march_30/ml.ADRP-ADPR_pocket1_dock.parquet --n_splits 3 --lc_sizes 4 --max_size 185000 --gout ./trn_lc_batch_serial_fly --trg_name reg --par_jobs 1
```

3. Parallel execusion (generates train/val/test splits on the fly; `n_splits` parallel runs). Call `src/batch_lc.py` with `par_jobs` set to >1.
```
$ python src/batch_lc.py --datapath data/docking_data_march_30/ml.ADRP-ADPR_pocket1_dock.parquet --n_splits 3 --lc_sizes 4 --max_size 185000 --gout ./trn_lc_batch_parallel_fly --trg_name reg --par_jobs 3
```

4. Parallel execusion (use pre-computed data splits; need to provide the path to the splits; `n_splits` parallel runs). Call `src/batch_lc.py` with `par_jobs` set to >1.
```
$ python src/batch_lc.py --datapath data/docking_data_march_30/ml.ADRP-ADPR_pocket1_dock.parquet --splitdir data/docking_data_march_30/ml.ADRP-ADPR_pocket1_dock.splits --n_splits 3 --lc_sizes 4 --gout ./trn_lc_batch_parallel_splits --trg_name reg --par_jobs 3
```

5. Aggregate results from parallel execusion and plot learning curves:
```
$ python src/agg_results_from_runs.py --res_dir ./trn_lc_batch_parallel_fly
```
```
$ python src/agg_results_from_runs.py --res_dir ./trn_lc_batch_parallel_splits
```

## Generate learning curves with your own ML model
If you want to use an ML model of your choice with the learning curve API, the minimum you need to provide is a function that generates your ML model and two python dictionaries.
One dict lists the model initialization parameters (`ml_init_kwargs`), and the other dict, contains the fitting (training) parameters (`ml_fit_kwargs`). For Keras model, you can also pass a function that creates a list of callbacks.<br>
See examples below. <be>

### LightGBM
```python
import lightgbm as lgb

# Define ML model
ml_model_def = lgb.LGBMRegressor

# Model init parameters
ml_init_kwargs = { 'n_estimators': 100, 'max_depth': -1,
	     	   'learning_rate': 0.1, 'num_leaves': 31,
	     	   'n_jobs': 8, 'random_state': 42 }

# Model fit parameters
ml_fit_kwargs = {'verbose': False, 'early_stopping_rounds': 10}
```

### Keras (TensorFlow)
```python
from pathlib import Path
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

def my_keras_model_def( input_dim ):
    """ Define keras model. """
    inputs = Input(shape=(input_dim,))
    x = Dense(250, activation='relu')(inputs)
    x = Dense(125, activation='relu')(x)
    x = Dense(60, activation='relu')(x)
    outputs = Dense(1, activation='relu')(x)
    model = Model(inputs=inputs, outputs=outputs)

    opt = SGD(lr=0.0001, momentum=0.9)
    model.compile(loss='mean_squared_error',
                  optimizer=opt, metrics=['mae'])
    return model

def my_callback_def(outdir, ref_metric='val_loss'):
    """ Define keras callbacks list. """
    checkpointer = ModelCheckpoint( str(outdir/'model_best.h5'), monitor='val_loss',
                                    save_weights_only=False, save_best_only=True )
    csv_logger = CSVLogger( outdir/'training.log' )
    reduce_lr = ReduceLROnPlateau( monitor=ref_metric, factor=0.75, patience=25 )
    early_stop = EarlyStopping( monitor=ref_metric )
    return [checkpointer, csv_logger, early_stop, reduce_lr]

ml_model_def = my_keras_model_def
keras_callbacks_def = my_callback_def

ml_init_kwargs = {'input_dim': xdata.shape[1]}   # Model init parameters
ml_fit_kwargs  = {'epochs': 300, 'batch_size': 32, 'verbose': 1}  # Model fit parameters

```
