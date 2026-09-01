
import tkinter as tk
import numpy as np
from customtkinter import *
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


set_appearance_mode("dark")
set_default_color_theme("dark-blue")

Wind = CTk()
Wind.geometry("500x700")
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
        label1.configure(text="Invalid ticker or no data found.")
        label2.configure(text="")
        label3.configure(text="")
        label4.configure(text="")
        label5.configure(text="")
        label6.configure(text="")
        label7.configure(text="")
        label8.configure(text="")
        label9.configure(text="")
        return

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["SMA50"] = data["Close"].rolling(window=50).mean()
    data["EMA20"] = data["Close"].ewm(span=20,adjust=False).mean()

    dif = data["Close"].diff()
    gain = dif.clip(lower=0).rolling(window=14).mean()
    loss = (-dif.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # changes between the days
    data["pct_change"] = data["Close"].pct_change()

    data["Volatility"] = data["pct_change"].rolling(window=20).std()

    data["Tomorrow"] = data["Close"].shift(-1)

    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

    features = ["SMA20","SMA50","EMA20","RSI","Volume","Volatility","pct_change"]

    latest = data[features].tail(1)

    newestmarketdata = data.tail(1)[
        ["Close", "SMA20", "SMA50", "EMA20", "RSI"]
    ]

    data.dropna(inplace=True)

    x = data[features]
    y = data["Target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        shuffle=False
    )


    combo1 = modelselector.get()

    if combo1 == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42
        )
    if combo1 == "Logistic Regression":
        model = LogisticRegression(
            max_iter=1000
        )
    if combo1 == "Neural Networks":
        model = MLPClassifier(
            hidden_layer_sizes=(50, 25),
            max_iter=1000,
            random_state=42
        )
    if combo1 == "Gradient Boosting":
        model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42
        )
    if combo1 == "Hist Gradient Boosting":
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42
        )
    if combo1 == "XGBoost":
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test,predictions)

    prediction = model.predict(latest)[0]
    probability = model.predict_proba(latest)[0]

    label1.pack(pady=2)
    label2.pack(pady=2)
    label3.pack(pady=2)
    label4.pack(pady=2)
    label5.pack(pady=5)
    label6.pack(pady=2)
    label7.pack(pady=2)
    label8.pack(pady=2)
    label9.pack(pady=2)

    label1.configure(text=f"Stock: {ticker}",font=("Times New Roman", 16, "bold"))

    label2.configure(text=f"Prediction Accuracy: {accuracy:.2%}",font=("Times New Roman", 16, "bold"))

    label3.configure(text="Latest Prediction:",font=("Times New Roman", 16, "bold"))

    if prediction == 1:
        label4.configure(text="Stock is likely to go UP tomorrow.",font=("Times New Roman", 15, "bold"))
    else:
        label4.configure(text="Stock is likely to go DOWN tomorrow.",font=("Times New Roman", 15, "bold"))


    label6.configure(text=f"Probability UP:   {probability[1]:.2%}",font=("Times New Roman", 14, "bold"))
    label7.configure(text=f"Probability DOWN: {probability[0]:.2%}",font=("Times New Roman", 14, "bold"))


    if probability[1] >= 0.60:
        signal = "BUY / KEEP SIGNAL"
    elif probability[1] <= 0.40:
        signal = "SELL SIGNAL"
    else:
        signal = "KEEP / HOLD"

    label5.configure(text=f"Signal: {signal}",font=("Times New Roman", 15, "bold"))
    label8.configure(text="Latest Market Data",font=("Times New Roman", 16, "bold"))
    label9.configure(text=newestmarketdata,font=("Times New Roman", 13))

#starting page

title = CTkLabel(Wind,text="STOCK PREDICTOR",font=("Times New Roman", 35, "bold"))
title.pack(pady=(5,5))


subtitle = CTkLabel(Wind,text="Machine Learning Market Prediction",font=("Times New Roman", 16))
subtitle.pack(pady=(0, 5))

ticker_label = CTkLabel(Wind,text="ENTER STOCK TICKER",font=("Times New Roman", 14, "bold"))
ticker_label.pack(pady=(10, 8))


stockname = CTkEntry(
    Wind,
    width=300,
    height=40,
    font=("Times New Roman", 14),
    justify="center",
    placeholder_text="Example: AAPL"
)
stockname.pack(pady=(0, 20))


model_label = CTkLabel(Wind,text="SELECT MACHINE LEARNING MODEL",font=("Times New Roman", 14, "bold"))

model_label.pack(pady=(0, 8))


items = (
    "Logistic Regression",
    "Random Forest",
    "Neural Networks",
    "Gradient Boosting",
    "Hist Gradient Boosting",
    "XGBoost"
)

modelselector = CTkComboBox(
    Wind,
    values=items,
    width=300,
    height=40,
    state="readonly",
    font=("Times New Roman", 14),
    justify="center"
)
modelselector.pack(pady=(0, 25))
modelselector.set("Select Machine Learning Model")


entry_button = CTkButton(
    Wind,
    text="PREDICT STOCK",
    command=stockticker,
    width=220,
    height=45,
    font=("Times New Roman", 15, "bold")
)
entry_button.pack(pady=(0, 15))


info = CTkLabel(Wind,text="Enter a ticker such as AAPL, TSLA, MCD or NVDA",font=("Times New Roman", 10))
info.pack(pady=(0, 15))

label1 = CTkLabel(Wind, text='')
label2 = CTkLabel(Wind, text='')
label3 = CTkLabel(Wind, text='')
label4 = CTkLabel(Wind, text='')
label5 = CTkLabel(Wind, text='')
label6 = CTkLabel(Wind, text='')
label7 = CTkLabel(Wind, text='')
label8 = CTkLabel(Wind, text='')
label9 = CTkLabel(Wind, text='')


Wind.mainloop()

