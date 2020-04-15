import numpy as np
import pandas as pd

def to_lstm_input(time_step, dataset, target_col_idx):
    X_train = []
    y_train = []

    for i in range(time_step, len(dataset)):
        X_train.append(dataset[i-time_step:i, :])
        y_train.append(dataset[i, target_col_idx])
        
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    return X_train, y_train

def predict_data(model, min_max_scaler, time_step, train_df, test_df):
    target_col_idx = 3
    # get the real price data
    real_stock_price = test_df.values[:, target_col_idx]
    real_stock_price_scaler = min_max_scaler.transform(test_df.values)
    
    # arrange the data to lstm input
    dataset_total = pd.concat([train_df, test_df], axis=0)
    testing_inputs = dataset_total[len(dataset_total)-len(test_df)-time_step:].values
    testing_scaled = min_max_scaler.transform(testing_inputs)
    X_test, _ = to_lstm_input(time_step, testing_scaled, target_col_idx)
    predicted_price_scaler = model.predict(X_test)
    
    # 將預測之 price_scaler 轉換回去
    real_stock_price_scaler[:, target_col_idx] = predicted_price_scaler.flatten()
    predicted_price = min_max_scaler.inverse_transform(real_stock_price_scaler)[:, target_col_idx]
    
    return real_stock_price, predicted_price