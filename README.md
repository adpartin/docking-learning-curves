The leaning curve methods currently support scikit-learn, LightGBM, and tf.keras ML models.

## Examples
Running from command line:
```
python src/main_lc.py --datapath data/dataframe.csv --n_shards 7 --n_splits 3 --gout ./trn_lc --trg_name reg
```

## 
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
