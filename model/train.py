from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


def train_lstm(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 6)))
    # model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences = True))
    # model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences = True))
    # model.add(Dropout(0.2))

    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

    # Adding the output layer
    model.add(Dense(units = 1))

    # Compiling
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # train the model
    model.fit(X_train, y_train, epochs = 16, batch_size = 16)
    
    return model