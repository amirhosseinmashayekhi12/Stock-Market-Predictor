# Stock-Market-Predictor
A machine learning project that analyzes historical stock market data and predicts whether a stock's price will move up or down the next trading day.
The program takes historical stock data and uses it to calculate different features, such as moving averages, RSI, daily returns, volume, and volatility. These features are then used to train a machine learning model.

After training, the model is tested on data it has not seen before. The program then uses the most recent stock data to make a prediction and shows the result through a simple graphical interface.

# How It Works

1. Historical stock data is collected using yFinance.
2. The data is cleaned and prepared using Pandas and NumPy.
3. Indicators and other features are calculated.
4. The data is split into training and testing.
5. A machine learning model is trained using the historical data.
6. The model is tested to see how well it performs.
7. The latest data is given to the model to predict the next day's direction.
8. The prediction and probability are displayed in the GUI.

# Technologies Used : 

- Python
- Pandas
- NumPy
- yFinance
- Scikit-learn
- XGBoost
- Tkinter
- Custom tkinter
- ttkbootstrap
  
# Goal

The goal of this project was to learn how machine learning can be used with real-world financial data and to see how well a model can predict short-term stock movement.
