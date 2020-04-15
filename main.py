import numpy as np
import pandas as pd

from tqdm import tqdm
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

from preprocess.build_lstm_input import to_lstm_input
from preprocess.get_stock_data import get_data_from_yahoo
from evaluation.evaluate import predict_data
from evaluation.plot_result import plot_result
from model.train import train_lstm

if __name__ == "__main__":
    # get data from Yahoo Finance
    tickers = ['AAPL']
    start_date = dt.datetime(2015, 1, 1)
    end_date = dt.datetime(2019, 12, 31)
    get_data_from_yahoo(tickers, start_date, end_date)

    ticker = tickers[0]
    comp_df = pd.read_csv('data/stock_data/{}.csv'.format(ticker))

    comp_df.set_index('Date', inplace=True)

    # 80% for training, 20% for testing
    # Cause the data is time-series -> not using train_test_split from sklearn
    train_stock_df, test_stock_df = comp_df['2015-01-07': '2019-01-02'], \
                                            comp_df['2019-01-03': '2019-11-26']

    
    # normalization
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    training_scaled = min_max_scaler.fit_transform(train_stock_df.values)

    # hyperparameters
    time_step = 180
    target_col_idx = 3 # 估計 close price -> 第三個 column

    # prepare lstm input -> np.array type
    X_train, y_train = to_lstm_input(time_step, training_scaled, target_col_idx)

    # train_lstm
    model = train_lstm(X_train, y_train)

    # predict stock price
    real_stock_price, predicted_price = predict_data(model, min_max_scaler, time_step, \
            train_stock_df, test_stock_df)
    
    # plot result
    plot_result(ticker, real_stock_price, predicted_price)
        
