import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from utils import CNNModel, LSTMModel, HyperParameterTuner

PROJECT_DIR = Path(__file__).resolve().parents[2]

BITSTREAM_CHUNKSIZE = 1
TEST_SIZE = 0.2
BATCH_SIZE = 256

def load_data(filepath: str, chunksize: int = 1, nrows = None):
    df = pd.read_csv(filepath, index_col=0, header=0, nrows=nrows)
    df['toolnr'] = pd.Categorical(df['tool']).codes
    X = np.array(df.iloc[:, :-2]).reshape(-1, int(8000/chunksize), chunksize)
    y = np.array(df.iloc[:, -1])
    num_classes = len(np.unique(y))
    return X, y, num_classes

def main():
    # Loading data
    X, y, num_classes = load_data(f"{PROJECT_DIR}/data/processed/merged_full.csv", nrows=10000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    # Build the model
    model = CNNModel(input_shape=(8000, 1), num_classes=num_classes)

    # Define the tuner
    tuner = HyperParameterTuner(model, PROJECT_DIR)

    # Run the search
    tuner.tune(X_train, y_train, X_test, y_test, 
               method="BayesianOptimization", trials=10)

if __name__ == "__main__":
    main()