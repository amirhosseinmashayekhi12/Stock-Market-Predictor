import tkinter as tk
import numpy as np
from tkinter import ttk
import ttkbootstrap as tk
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Wind= tk.Window(themename = 'cyborg')
Wind.geometry("500x500")
Wind.title("Machine Learning Market Predictor")



def stockticker():
    ticker = stockname.get().upper()
    data = yf.download(
        ticker,
        period="3y",
        auto_adjust=False,
        progress=False
    )
    if data.empty:
        label1.config(text="Invalid ticker or no data found.")
        label2.config(text="No data found.")
        label3.config(text="No data found.")
        label4.config(text="No data found.")
        label5.config(text="No data found.")
        label6.config(text="No data found.")
        label7.config(text="No data found.")
        label8.config(text="No data found.")
        label9.config(text="No data found.")
        return
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["SMA40"] = data["Close"].rolling(window=40).mean()

    data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()

    dif = data["Close"].diff()

    gain = dif.clip(lower=0).rolling(window=14).mean()
    loss = (-dif.clip(upper=0)).rolling(window=14).mean()

    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    data["pct_change"] = data["Close"].pct_change()

    data["Volatility"] = data["pct_change"].rolling(window=20).std()

    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

    features = [
        "SMA20",
        "SMA40",
        "EMA20",
        "RSI",
        "Volume",
        "Volatility",
        "pct_change",
    ]

    latest = data[features].tail(1)
    latest_market_data = data.tail(1)[
        ["Close", "SMA20", "SMA40", "EMA20", "RSI"]
    ]

    data.dropna(inplace=True)

    x = data[features]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        shuffle=False
    )
    combo1 = modelselector.get()
    if combo1 == 'Random Forest' :
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42
        )
    elif combo1 == 'Logistic Regression' :
        model = LogisticRegression()
    elif combo1 == 'Neural Networks' :
        model = MLPClassifier(
        hidden_layer_sizes=(50, 25),
        max_iter=1000,
        random_state=42
    )
    elif combo1 == 'Gradient Boosting' :
        model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42
    )
    elif combo1 == 'Hist Gradient Boosting':
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42
        )
    elif combo1 == 'XGBoost':
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )


    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    label1.config(text = f"Stock: {ticker}", font='Times 14 bold')
    label2.config(text = f"Prediction Accuracy: {accuracy:.2%}", font='Times 14 bold')

    prediction = model.predict(latest)[0]
    probability = model.predict_proba(latest)[0]

    label3.config(text = "\nLatest Prediction:", font='Times 14 bold')

    if prediction == 1:
        label4.config(text = "Stock is likely to go UP tomorrow.", font='Times 14 bold')
    else:
        label4.config(text = "Stock is likely to go DOWN tomorrow.", font='Times 14 bold')

    label6.config(text = f"Probability UP:   {probability[1]:.2%}", font='Times 14 bold')
    label7.config(text = f"Probability DOWN: {probability[0]:.2%}", font='Times 14 bold')

    if probability[1] >= 0.60:
        signal = "BUY/KEEP SIGNAL"
    elif probability[1] <= 0.40:
        signal = "SELL SIGNAL"
    else:
        signal = "KEEP / HOLD"

    label8.config(text = "\nLatest Market Data", font='Times 14 bold')
    label9.config(text = latest_market_data, font='Times 14 bold')
    label5.config(text = f"Signal: {signal}", font='Times 14 bold')

title = ttk.Label(
    Wind,
    text="STOCK PREDICTOR",
    font=("Times New Roman", 32, "bold"),
)
title.pack(pady=(35, 5))

subtitle = ttk.Label(
    Wind,
    text="Machine Learning Market Prediction",
    font=("Times New Roman", 16),
)
subtitle.pack(pady=(0, 25))

input_frame = ttk.Frame(Wind, padding=20)
input_frame.pack(pady=5)

ticker_label = ttk.Label(
    input_frame,
    text="ENTER STOCK TICKER",
    font=("Times New Roman", 14, "bold")
)
ticker_label.pack(pady=(0, 8))

stockname = ttk.Entry(
    input_frame,
    width=30,
    font=("Times New Roman", 14),
    justify="center"
)
stockname.pack(ipady=7, pady=(0, 20))
#stockname is our entry
model_label = ttk.Label(
    input_frame,
    text="SELECT MACHINE LEARNING MODEL",
    font=("Times New Roman", 14, "bold")
)
model_label.pack(pady=(0, 8))

items = (
    'Logistic Regression',
    'Random Forest',
    'Neural Networks',
    'Gradient Boosting',
    'Hist Gradient Boosting',
    'XGBoost'
)

modelselector = ttk.Combobox(
    input_frame,
    values=items,
    width=27,
    state="readonly",
    font=("Times New Roman", 14),
    justify="center"
)
modelselector.pack(ipady=6, pady=(0, 25))


modelselector.current(1)



style = ttk.Style()

style.configure(
    "Predict.TButton",
    font=("Times New Roman", 15, "bold"),
    padding=(20, 12)
)



entry_button = ttk.Button(
    input_frame,
    text="  PREDICT STOCK  ",
    command=stockticker,
    style="Predict.TButton"
)
entry_button.pack(pady=5)


info = ttk.Label(
    Wind,
    text="Enter a ticker such as AAPL, TSLA, MCD  or NVDA",
    font=("Times New Roman", 10)
)
info.pack(pady=(15, 20))



label1 = ttk.Label(Wind)
label1.pack()

label2 = ttk.Label(Wind)
label2.pack()

label3 = ttk.Label(Wind)
label3.pack()

label4 = ttk.Label(Wind)
label4.pack()

label5 = ttk.Label(Wind)
label5.pack()

label6 = ttk.Label(Wind)
label6.pack()

label7 = ttk.Label(Wind)
label7.pack()

label8 = ttk.Label(Wind)
label8.pack()

label9 = ttk.Label(Wind)
label9.pack()


Wind.mainloop()