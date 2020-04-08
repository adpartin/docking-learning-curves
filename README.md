The leaning curve methods currently support scikit-learn, LightGBM, and tf.keras ML models.

## Examples
Running from command line:
```
python src/main_lc.py --datapath data/dataframe.csv
```

## 
If you want to use an ML model of your choice with the learning curve API, you need to provide a function that generates the ML model and two python dictionaries.
One dict provides the model initialization model parameters (ml_init_kwargs), and the other, the fitting (training) parameters (ml_fit_kwargs). For Keras model, you can also pass a function that defines a callback list callback list.<be>
For example:

### LightGBM
```python
# Define ML model
ml_model_def = lgb.LGBMRegressor

# Model init parameters
ml_init_kwargs = { 'n_estimators': 100, 'max_depth': -1,
	     	   'learning_rate': 0.1, 'num_leaves': 31,
	     	   'n_jobs': 8, 'random_state': 42 }

# Model fit parameters
ml_fit_kwargs = {'verbose': False, 'early_stopping_rounds': 10}
```

### TF Keras
```python
from pathlib import Path
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Define ML model
def my_keras_model_def( input_dim ):
    """ Define keras model. """
    inputs = Input(shape=(input_dim,))
    x = Dense(250, activation='relu')(inputs)
    x = Dense(125, activation='relu')(x)
    x = Dense(60, activation='relu')(x)
    outputs = Dense(1, activation='relu')(x)
    model = Model(inputs=inputs, outputs=outputs)

    opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    opt = SGD(lr=0.0001, momentum=0.9)
    model.compile(loss='mean_squared_error',
                  optimizer=opt, metrics=['mae'])
    return model

ml_model_def = my_keras_model_def

# Define callback list
def my_callback_def(outdir, ref_metric='val_loss'):
    """ Define keras callbacks list. """
    checkpointer = ModelCheckpoint( str(outdir/'model_best.h5'), monitor='val_loss', verbose=0,
                                    save_weights_only=False, save_best_only=True )
    csv_logger = CSVLogger( outdir/'training.log' )
    reduce_lr = ReduceLROnPlateau( monitor=ref_metric, factor=0.75, patience=25, verbose=1,
                                   mode='auto', min_delta=0.0001, cooldown=3, min_lr=0.000000001 )
    early_stop = EarlyStopping( monitor=ref_metric, patience=50, verbose=1, mode='auto' )
    return [checkpointer, csv_logger, early_stop, reduce_lr]

keras_callbacks_def = my_callback_def

# Model init parameters
ml_init_kwargs = {'input_dim': xdata.shape[1], 'dr_rate': 0.1}

# Model fit parameters
ml_fit_kwargs = {'epochs': 300, 'batch_size': 32, 'verbose': 1}

```
