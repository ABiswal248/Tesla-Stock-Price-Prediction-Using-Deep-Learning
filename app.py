import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px


from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Tesla Stock Prediction",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("Tesla Stock Price Prediction")
st.markdown("### Deep Learning Based Stock Forecasting using GRU")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model("models/final_gru_model.keras")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Tesla Dataset (TSLA.csv)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # -----------------------------
    # DATASET PREVIEW
    # -----------------------------
    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # -----------------------------
    # DATASET INFORMATION
    # -----------------------------
    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    # -----------------------------
    # CLOSE PRICE GRAPH
    # -----------------------------
    st.subheader("Tesla Closing Price Trend")

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(df["Date"])

        fig = px.line(
            df,
            x="Date",
            y="Close",
            title="Tesla Closing Price Over Time"
        )

        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # MODEL COMPARISON
    # -----------------------------
    st.subheader("Model Performance Comparison")

    comparison = pd.DataFrame({
        "Model": ["SimpleRNN", "LSTM", "GRU"],
        "MSE": [1105.18, 682.25, 393.31],
        "RMSE": [33.24, 26.12, 19.83],
        "MAE": [23.77, 16.93, 13.26]
    })

    st.dataframe(comparison)

    st.success(
        "Best Performing Model: GRU"
    )

    # -----------------------------
    # DATA PREPARATION
    # -----------------------------
    close_data = df[['Close']]

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(close_data)

    # -----------------------------
    # PREDICTION OPTIONS
    # -----------------------------
    st.subheader("Future Prediction")

    horizon = st.selectbox(
        "Select Prediction Horizon",
        [1, 5, 10]
    )

    # -----------------------------
    # PREDICT BUTTON
    # -----------------------------
    if st.button("Predict Stock Price"):

        future_input = scaled_data[-60:].copy()

        predictions = []

        for i in range(horizon):

            pred = model.predict(
                future_input.reshape(1, 60, 1),
                verbose=0
            )

            predictions.append(pred[0][0])

            future_input = np.append(
                future_input[1:],
                pred
            )

        predictions = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )

        prediction_df = pd.DataFrame({
            "Day": range(1, horizon + 1),
            "Predicted Price": predictions.flatten()
        })

        # -----------------------------
        # SHOW TABLE
        # -----------------------------
        st.subheader("Predicted Prices")

        st.dataframe(prediction_df)

        # -----------------------------
        # SHOW GRAPH
        # -----------------------------
        st.subheader("Forecast Visualization")

        fig2 = px.line(
            prediction_df,
            x="Day",
            y="Predicted Price",
            markers=True,
            title="Future Tesla Stock Price Prediction"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # -----------------------------
        # SUMMARY
        # -----------------------------
        st.subheader("Prediction Summary")

        st.write(
            f"""
            Using the GRU Deep Learning model,
            the Tesla stock price has been forecasted
            for the next **{horizon} day(s)**.
            """
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "Developed using Deep Learning Models: SimpleRNN, LSTM and GRU"
)