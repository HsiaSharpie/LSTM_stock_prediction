import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import pandas_datareader.data as web


def get_data_from_yahoo(tickers, start, end):
    if not os.path.exists('data/stock_data'):
        os.mkdir('data/stock_data')

    for ticker in tickers:
        if not os.path.exists('stock_data/{}'.format(ticker)):
            try:
                df = web.DataReader(ticker, 'yahoo', start, end)
                df.to_csv('data/stock_data/{}.csv'.format(ticker))
            except:
                continue
        else:
            print('Already have {}'.format(ticker))