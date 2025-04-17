import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

weather_data = pd.read_csv('Anomaly Model/processed_weather_data.csv')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(weather_data.drop(columns=['TimeStamp']))

look_back = 12  # Use 12 months of data to predict the next month
X, y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i-look_back:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=y.shape[1]))


model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=50, batch_size=32)

model.save('Anomaly Model/lstm_model.h5')
joblib.dump(scaler, 'Anomaly Model/scaler.pkl')

print("LSTM model and scaler saved.") 