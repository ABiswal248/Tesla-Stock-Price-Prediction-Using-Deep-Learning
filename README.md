# Tesla Stock Price Prediction Using Deep Learning

## Project Overview

This project focuses on predicting Tesla stock closing prices using Deep Learning techniques. The objective is to analyze historical Tesla stock market data and develop predictive models capable of forecasting future stock prices.

Three different recurrent neural network architectures were implemented and compared:

- SimpleRNN
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

The models were trained using historical Tesla stock price data and evaluated using multiple performance metrics. The best-performing model was deployed through a Streamlit web application to provide future stock price predictions.

---

## Problem Statement

Stock prices are sequential in nature and depend on historical patterns over time. Traditional machine learning models often struggle to capture long-term temporal dependencies present in financial time-series data.

The objective of this project is to:

- Analyze Tesla stock price data.
- Perform data cleaning and preprocessing.
- Create time-series sequences for forecasting.
- Build and compare SimpleRNN, LSTM, and GRU models.
- Predict future stock prices for 1-day, 5-day, and 10-day horizons.
- Deploy the best-performing model using Streamlit.

---

## Dataset Information

The dataset contains historical Tesla stock market information with the following attributes:

| Column    | Description              |
| --------- | ------------------------ |
| Date      | Trading Date             |
| Open      | Opening Price            |
| High      | Highest Price of the Day |
| Low       | Lowest Price of the Day  |
| Close     | Closing Price            |
| Adj Close | Adjusted Closing Price   |
| Volume    | Number of Shares Traded  |

The closing price was selected as the target variable for prediction.

---

## Project Workflow

### 1. Data Collection

- Historical Tesla stock data was collected.
- Dataset loaded using Pandas.

### 2. Data Cleaning

- Checked for missing values.
- Verified duplicate records.
- Corrected data types.
- Converted Date column into datetime format.

### 3. Exploratory Data Analysis

Several visualizations were performed:

- Tesla Closing Price Trend
- Trading Volume Analysis
- Historical Price Behaviour
- Feature Correlation Analysis

### 4. Data Preprocessing

- Selected Closing Price as the target feature.
- Applied MinMaxScaler normalization.
- Created time-series sequences using a rolling window approach.
- Split the data into training and testing datasets.

### 5. Model Development

Three Deep Learning models were developed:

#### SimpleRNN

A basic recurrent neural network architecture used as the baseline model.

#### LSTM

An advanced recurrent neural network architecture designed to capture long-term dependencies in sequential data.

#### GRU

A computationally efficient recurrent architecture capable of capturing temporal dependencies while reducing training complexity.

### 6. Model Evaluation

The models were evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

---

## Model Performance

| Model     | MSE     | RMSE  | MAE   |
| --------- | ------- | ----- | ----- |
| SimpleRNN | 1105.18 | 33.24 | 23.77 |
| LSTM      | 682.25  | 26.12 | 16.93 |
| GRU       | 393.31  | 19.83 | 13.26 |

### Best Performing Model

GRU achieved the lowest prediction error and therefore was selected as the final deployment model.

---

## Future Price Prediction

The deployed model supports:

- 1-Day Forecast
- 5-Day Forecast
- 10-Day Forecast

Predictions are generated using the most recent historical observations and the trained GRU model.

---

## Streamlit Application Features

The web application provides:

- Dataset Upload Functionality
- Dataset Preview
- Dataset Statistics
- Tesla Stock Price Visualization
- Model Performance Comparison
- Future Price Prediction
- Interactive Forecast Graphs

---

## Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- TensorFlow
- Keras
- Streamlit

### Development Tools

- Visual Studio Code
- Jupyter Notebook
- Git
- GitHub

---

## Project Structure

```text
Tesla-Stock-Price-Prediction-Using-Deep-Learning
│
├── data
│   └── TSLA.csv
│
├── models
│   ├── best_rnn.keras
│   ├── best_lstm.keras
│   ├── best_gru.keras
│   └── final_gru_model.keras
│
├── notebook
│   └── Tesla_Stock_Prediction.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ABiswal248/Tesla-Stock-Price-Prediction-Using-Deep-Learning.git
```

Navigate to the project directory:

```bash
cd Tesla-Stock-Price-Prediction-Using-Deep-Learning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Conclusion

This project demonstrates the application of Deep Learning techniques for financial time-series forecasting. Three recurrent neural network architectures were implemented and compared. Experimental results showed that the GRU model achieved the highest predictive performance on Tesla stock data and was selected for deployment.

The project provides a practical example of combining data analysis, deep learning, and web deployment to build an end-to-end stock price forecasting solution.

---

## Future Enhancements

- Integration of real-time stock market data.
- Incorporation of news sentiment analysis.
- Addition of macroeconomic indicators.
- Comparison with Transformer-based architectures.
- Multi-stock forecasting support.
- Cloud deployment and automated model retraining.
