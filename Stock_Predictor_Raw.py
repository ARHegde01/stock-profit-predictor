import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from collections import OrderedDict
import matplotlib.pyplot as plt

ticker = input("Please enter the stock ticker symbol (Please include ^ for market indices): ")
sp500 = yf.Ticker(f"{ticker}")
sp500 = sp500.history(period="max")
sp500.plot.line(y="Close", use_index=True)

if "Dividends" in sp500.columns:
    del sp500["Dividends"]
if "Stock Splits" in sp500.columns:
    del sp500["Stock Splits"]

sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["2003-01-01":].copy()
sp500

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .60] = 1
    preds[preds < .60] = 0
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined

def backtest(data, model, predictors, start=250, step=63):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i: (i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]
predictors = ["Close", "Volume", "High", "Low"]
model.fit(train[predictors], train["Target"])
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()

predictions = backtest(sp500, model, predictors)

horizons = [2, 5, 60, 250, 2000]
new_predictors = []
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    new_predictors += [ratio_column, trend_column]

sp500.dropna(inplace=True)

# Check if there are enough samples for training
if sp500.shape[0] > 100:
    train = sp500.iloc[:-100].copy()
    test = sp500.iloc[-100:].copy()
    predictors = ["Close", "Volume", "High", "Low"]
    target_column = "Target"
    if train[predictors].shape[0] > 0 and train[target_column].shape[0] > 0:
        model = RandomForestClassifier(n_estimators=200, min_samples_split=25, random_state=1)
        model.fit(train[predictors], train[target_column])
        preds = model.predict(test[predictors])
        # Further analysis or evaluation here
        
        # Calculate precision score
        precision = precision_score(test[target_column], preds)
        print(f"Precision Score: {precision}")
        
        # Plot predicted vs actual values
        plt.figure(figsize=(12, 6))
        plt.plot(test.index, test[target_column], label="Actual")
        plt.plot(test.index, preds, label="Predicted", linestyle='dashed')
        plt.legend()
        plt.show()

    else:
        print("Training data is empty. Check your data preprocessing and slicing.")
else:
    print("Not enough samples in the dataset for training and testing.")

investment_horizons = [10, 20, 63]  # 2 weeks, 1 month, 1 quarter

for horizon in investment_horizons:
    target_column = f"Target_{horizon}"
    sp500.loc[:, target_column] = (sp500["Close"].shift(-horizon) > sp500["Close"]).astype(int)

    investment_horizons = [10, 20, 63]  # 2 weeks, 1 month, 1 quarter
# Store the likelihoods for each time frame
profit_likelihoods = OrderedDict({
    "2_weeks": None,
    "1_month": None,
    "1_quarter": None
})
time_frames = list(profit_likelihoods.keys())

threshold = 0.5 # Adjust this threshold as needed

for idx, horizon in enumerate(investment_horizons):
    target_column = f"Target_{horizon}"
    train = sp500.iloc[:-100].copy()
    test = sp500.iloc[-100:].copy()
    model = RandomForestClassifier(n_estimators=200, min_samples_split=25, random_state=1)
    model.fit(train[new_predictors], train[target_column])
    preds_proba = model.predict_proba(test[new_predictors])[:,1]

    # Calculate the likelihood of profit for the corresponding time frame
    profit_likelihoods[time_frames[idx]] = (preds_proba > threshold).mean()

# Output the likelihood of profit
for time_frame, likelihood in profit_likelihoods.items():
    print(f"The likelihood of making a profit in the next {time_frame} is {likelihood * 100}%.")
