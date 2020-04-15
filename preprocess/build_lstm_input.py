import numpy as np

def to_lstm_input(time_step, dataset, target_col_idx):
    X_train = []
    y_train = []

    for i in range(time_step, len(dataset)):
        X_train.append(dataset[i-time_step:i, :])
        y_train.append(dataset[i, target_col_idx])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    return X_train, y_train