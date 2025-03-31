# Big Data Engineering: S&P500 Price Prediction

## Overview

This project predicts S&P500 stock prices using big data and AI techniques. It streams financial data and economic indicators through Apache Kafka, stores the information in MongoDB Atlas, and trains LSTM models to forecast stock prices up to three days ahead. A Streamlit dashboard then visualizes historical data alongside model predictions in real time.

## Motivation

Accurate stock price forecasting is crucial for investors and analysts. By leveraging real-time data streaming and machine learning, this project demonstrates how to integrate big data architectures with predictive analytics to support informed decision-making in financial markets.

## Architecture & Setup

- **Data Streaming:**  
  - **Apache Kafka:** Used for streaming continuous S&P500 index data and various economic indicators.  
  - **Kafka Topics:** Dedicated topics are created (e.g., `sp500`, `sp500_pred`, `UNRATE`, etc.) to manage different data streams.

- **Data Storage:**  
  - **MongoDB Atlas:** A cloud-based NoSQL database stores the streamed data in a database named "Stock" with collections for each topic.

- **Predictive Modeling:**  
  - **LSTM Models:** Three LSTM models are trained using windowed time-series data to predict the S&P500 closing price for the next 1, 2, and 3 days.  

- **Visualization:**  
  - **Streamlit Dashboard:** Provides real-time visualizations of historical and predicted stock prices, complete with filtering options and trend charts.
