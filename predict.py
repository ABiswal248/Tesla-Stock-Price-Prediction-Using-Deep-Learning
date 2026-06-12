import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model = load_model(
    "models/final_gru_model.keras"
)

df = pd.read_csv(
    "data/TSLA.csv"
)

close_data = df[['Close']]

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    close_data
)

future_input = scaled_data[-60:].copy()

prediction = model.predict(
    future_input.reshape(1,60,1)
)

prediction = scaler.inverse_transform(
    prediction
)

print(
    "Tomorrow Predicted Price:"
)

print(
    prediction[0][0]
)