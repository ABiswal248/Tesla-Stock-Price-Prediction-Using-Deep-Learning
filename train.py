import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

print("Loading Dataset...")

df = pd.read_csv("data/TSLA.csv")

close_data = df[['Close']]

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(close_data)

window_size = 60

X = []
y = []

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

model = Sequential()

model.add(
    GRU(
        units=50,
        return_sequences=False,
        input_shape=(60,1)
    )
)

model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mse'
)

print("Training Model...")

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32
)

model.save(
    "models/final_gru_model.keras"
)

print("Model Saved Successfully")