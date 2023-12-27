# Short-Term-Memory-LSTM-
Build an LSTM-based neural network for predicting stock prices using historical data.
# Implementing an LSTM model for stock price prediction (example code)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess historical stock data
# Assume 'df' is a DataFrame with columns 'Date' and 'Close'

scaler = MinMaxScaler(feature_range=(0, 1))
df['Close'] = scaler.fit_transform(np.array(df['Close']).reshape(-1, 1))

# Create sequences for training the LSTM model
sequence_length = 10
sequences = []
target = []

for i in range(len(df) - sequence_length):
    sequences.append(df['Close'].iloc[i:i+sequence_length].values)
    target.append(df['Close'].iloc[i+sequence_length])

X = np.array(sequences)
y = np.array(target)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=20, batch_size=32)

# Make predictions
test_sequences = df['Close'].values[-sequence_length:]
test_sequences = scaler.transform(test_sequences.reshape(-1, 1))
test_sequences = np.reshape(test_sequences, (1, sequence_length, 1))

predicted_price = model.predict(test_sequences)
predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))

print(f"Predicted Stock Price: {predicted_price[0][0]}")
