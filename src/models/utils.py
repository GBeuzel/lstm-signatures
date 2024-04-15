from keras import Input
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Bidirectional, LSTM, Dense, Reshape, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

import keras_tuner as kt
from names_generator import generate_name

PATIENCE = 3
EPOCHS = 25

class CNNModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(Reshape((1000, 8, 1)))

        for i in range(hp.Int('num_conv_layers', 1, 3)):
            model.add(Conv2D(hp.Int(f'conv_{i}_units', 32, 256, step=32), (3, 3), padding='same', activation='relu'))
            model.add(MaxPooling2D((2, 2)))

            if hp.Boolean(f'conv_{i}_dropout', default=False):
                model.add(Dropout(rate=hp.Float(f'conv_{i}_dropout_rate', 0.0, 0.5, step=0.1)))
        
        model.add(Flatten())

        if hp.Boolean(f'extra_dropout', default=False):
            model.add(Dropout(rate=hp.Float(f'conv_{i}_dropout_rate', 0.0, 0.5, step=0.1)))
        
        model.add(Dense(self.num_classes, activation='softmax'))
        opt = Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))
        model.compile(optimizer=opt, 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'],)
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int("batch_size", 32, 256, step=32, default=128),
            **kwargs,
        )

class LSTMModel(kt.HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=self.input_shape))

        if hp.Boolean("bidirectional"):
            model.add(Bidirectional(LSTM(hp.Int('input_unit', min_value=8, max_value=128, step=8), activation='tanh', return_sequences=True)))
        else:
            model.add(LSTM(hp.Int('input_unit', min_value=8, max_value=128, step=8), activation='tanh', return_sequences=True))

        if hp.Boolean("extra_lstm"):
            model.add((LSTM(hp.Int('input_unit', min_value=8, max_value=128, step=8), activation='tanh', return_sequences=True)))

        model.add(Dropout(hp.Float('dropout_rate', min_value=0, max_value=0.5, step=0.1)))
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
class HyperParameterTuner:
    def __init__(self, model, project_dir):
        self.model = model
        self.project_name = generate_name()
        self.tuner_dir = f'{project_dir}/models/logs'
        self.checkpoint_path = f'{project_dir}/models/weights/{self.project_name}/checkpoints.weights.h5'
        print(f"Starting a new project {self.project_name}! Good luck...")
    
    def tune(self, X_train, y_train, X_test, y_test, method="BayesianOptimization", trials=10):
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE),
            EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='auto', restore_best_weights=True),
            ModelCheckpoint(filepath=self.checkpoint_path, save_weights_only=True, verbose=1)]
        
        tuner_methods = {
            "BayesianOptimization": kt.BayesianOptimization,
            "RandomSearch": kt.RandomSearch,
            "Hyperband": kt.Hyperband}

        if method not in tuner_methods:
            raise ValueError(f"Invalid tuner method: {method}. Supported methods: {list(tuner_methods.keys())}")
        
        if method == 'Hyperband':
            tuner = tuner_methods[method](
                self.model,
                objective='val_accuracy',
                max_epochs=trials,
                hyperband_iterations=3,
                overwrite=True,
                project_name=self.project_name,
                directory= self.tuner_dir)
        else:
            tuner = tuner_methods[method](
                self.model,
                objective='val_accuracy',
                max_trials=trials,
                overwrite=True,
                project_name=self.project_name,
                directory= self.tuner_dir)

        tuner.search(X_train, y_train,
                     epochs=EPOCHS,
                     validation_data=(X_test, y_test),
                     callbacks=callbacks)
        print(tuner.results_summary())

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best hyperparameters: {best_hps}")

        best_model = tuner.get_best_models(num_models=1)[0]
        print(best_model.summary())

locals