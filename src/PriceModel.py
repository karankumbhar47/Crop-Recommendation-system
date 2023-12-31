from keras.layers import Dense, LSTM, Input
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from datetime import datetime
import pickle
import numpy as np
import pandas as pd

np.random.seed(42)


class PriceModel():
    def __init__(self,
                 trained: bool = True,
                 show_logs: bool = False,
                 sequence_length : int = 5,
                 batch_size: int = 16,
                 cross_val: bool = True,
                 save_as: str = None,
                 file: str = None):
        self.batch_size = batch_size
        self.cross_val = cross_val
        self.show_logs = show_logs
        self.sequence_length = sequence_length
        if trained == False:
            # load the processed data
            self.data = pd.read_csv(file)
            self.model = None
            self.train()
        else:
            self.model = load_model(file)
            self.scaler = pickle.load(open(f"{file}_scaler.pkl", "rb"))

    def train(self,
              save_as: str = None,
              epochs: int = 10) -> None:
        train_df = []
        test_df = []
        if (self.cross_val):
            split_date = np.percentile(self.data["arrival_date"], 90)
            if (self.show_logs):
                print(f"\tStart date {datetime.fromordinal(np.min(self.data['arrival_date']))}")
                print(f"\tSplit date {datetime.fromordinal(int(split_date))}")
                print(f"\tEnd date {datetime.fromordinal(np.max(self.data['arrival_date']))}")

            mask = self.data["arrival_date"] < split_date
            train_df = self.data[mask].copy()
            test_df = self.data[~mask].copy()
        else:
            train_df = self.data

        self.scaler = MaxAbsScaler()
        train_df[['min_price', 'max_price', 'modal_price']] = self.scaler.fit_transform(train_df[['min_price', 'max_price', 'modal_price']])
        if(self.cross_val):
            test_df[['min_price', 'max_price', 'modal_price']] = self.scaler.transform(test_df[['min_price', 'max_price', 'modal_price']])

        X_train, y_train = [], []
        for i in range(len(train_df) - self.sequence_length):
            X_train.append(train_df.iloc[i:i+self.sequence_length][['arrival_date', 'latitude', 'longitude']].values)
            y_train.append(train_df.iloc[i+self.sequence_length][['min_price', 'max_price', 'modal_price']].values)
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).astype('float32')

        X_test, y_test = [], []
        if(self.cross_val):
            for i in range(len(test_df) - self.sequence_length):
                X_test.append(test_df.iloc[i:i+self.sequence_length][['arrival_date', 'latitude', 'longitude']].values)
                y_test.append(test_df.iloc[i+self.sequence_length][['min_price', 'max_price', 'modal_price']].values)
            X_test = np.asarray(X_test)
            y_test = np.asarray(y_test).astype('float32')

        self.model = Sequential()
        self.model.add(LSTM(3, input_shape=(self.sequence_length, 3), batch_size=self.batch_size, kernel_regularizer='l2'))
        self.model.add(Dense(32, activation='tanh'))
        self.model.add(Dense(3, activation='linear'))

        self.model.compile(optimizer=Adam(), loss='mean_squared_error')

        self.model.fit(X_train, y_train,
                       epochs=epochs,
                       batch_size=self.batch_size,
                       validation_split=0.1)
        if self.cross_val:
            evaluation = self.model.evaluate(X_test, y_test)
            if self.show_logs:
                print(f"Mean Absolute Error on Test Set: {evaluation}")
        if save_as is not None:
            self.model.save(save_as)
            pickle.dump(self.scaler, open(f"{save_as}_scaler.pkl", "wb"))

    def predict(self, date: datetime, long: float, lat: float) -> np.array:
        predictDate = date.toordinal()
        print(predictDate)
        prediction = self.model.predict([[
            [predictDate - i, lat, long] for i in range(4,-1,-1)
        ]])
        prices = self.scaler.inverse_transform(prediction)
        return prices
