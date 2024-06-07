import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('energydata_complete.csv',index_col='date', infer_datetime_format=True)

df.head()

df.info()

df['Windspeed'].plot(figsize=(12,8))

df['Appliances'].plot(figsize=(12,8))

print(len(df))

print(df.head(3))

print(df.tail(5))

print(df.loc['2016-05-01':])

df = df.loc['2016-05-01':]

df = df.round(2)

print(len(df))

print(24*6)

test_day = 2

test_ind = test_day*144

print(test_ind)

train = df.iloc[:-test_ind]
test = df.iloc[-test_ind:]

print(train)

print(test)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(train)

scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

length = 144 # Length of the output sequences (in number of timesteps)
batch_size = 1 #Number of timeseries samples in each batch
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)

print(len(scaled_train))

print(len(generator))

X,y = generator[0]

print(f'Given the Array: \n{X.flatten()}')
print(f'Predicting this y: \n {y}')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

print(scaled_train.shape)

model = Sequential()

model.add(LSTM(100, input_shape=(length,scaled_train.shape[1])))

model.add(Dense(scaled_train.shape[1]))

model.compile(optimizer='adam', loss='mse')

model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=1)
validation_generator = TimeseriesGenerator(scaled_test, scaled_test, length=length, batch_size=batch_size)
model.fit_generator(generator,epochs=4, validation_data=validation_generator, callbacks=[early_stop])
model.history.history.keys()

losses = pd.DataFrame(model.history.history)
losses.plot()

first_eval_batch = scaled_train[-length:]

print(first_eval_batch)

first_eval_batch = first_eval_batch.reshape((1, length, scaled_train.shape[1]))

model.predict(first_eval_batch)

print(scaled_test[0])

n_features = scaled_train.shape[1]
test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(720):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

print(test_predictions)
print(scaled_test)
true_predictions = scaler.inverse_transform(test_predictions)
print(true_predictions)
print(test)
true_predictions = pd.DataFrame(data=true_predictions,columns=test.columns)
print(true_predictions.tail(432))
