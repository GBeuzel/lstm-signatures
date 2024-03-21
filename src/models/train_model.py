import os
import numpy as np
import pandas as pd
from pathlib import Path

from names_generator import generate_name

from sklearn.model_selection import train_test_split

from keras import Input
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Flatten, Dropout, Bidirectional, LSTM, Dense


BITSTREAM_CHUNKSIZE = 1
TEST_SIZE = 0.2
EPOCHS = 2
BATCH_SIZE = 256
PATIENCE = 10
MODEL_NAME = generate_name()

PROJECT_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = f'{PROJECT_DIR}/models/weights/{MODEL_NAME}/'
CHECKPOINT_PATH = f'{PROJECT_DIR}/models/weights/{MODEL_NAME}/checkpoint.weights.h5'
LOG_DIR = f'models/logs/fit/{MODEL_NAME}'

def load_data(filepath: str, chunksize: int = 1, nrows = None):
    df = pd.read_csv(filepath, index_col=0, header=0, nrows=nrows)
    df['toolnr'] = pd.Categorical(df['tool']).codes
    X = np.array(df.iloc[:, :-2]).reshape(-1, int(8000/chunksize), chunksize)
    y = np.array(df.iloc[:, -1])
    return X, y

def build_model(shape, num_classes):
    model = Sequential()
    model.add(Input(shape=shape))
    model.add(Bidirectional(LSTM(8, activation='relu', return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.name = MODEL_NAME
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_and_evaluate_model(model, 
                             X_train, y_train, X_test, y_test):
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE), 
                 ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1),
                 TensorBoard(log_dir=LOG_DIR, histogram_freq=1)]

    print(f"Training model...")
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks,
                        verbose=1)
    
    return history


## Load data
X, y = load_data(f"{PROJECT_DIR}/data/processed/merged_full.csv", nrows=10000)

num_classes = len(np.unique(y))
print(f"Unique classes: {np.unique(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
print((X_train.shape, y_train.shape, X_test.shape, y_test.shape))
print('-'*40)

# ## Build model
model = build_model(shape=(X.shape[1], X.shape[2],), num_classes=num_classes)
print('-'*40)

# ## Train and evaluate
os.mkdir(CHECKPOINT_DIR)
history = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
print('-'*40)