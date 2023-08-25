## Stock-Profit-Predictor

### The Stock Price Predictor is an advanced AI-powered program designed to analyze historical stock or market index data and predict whether the stock price will rise the next day. It employs the RandomForestClassifier for its predictions, a machine learning model that outputs predictions based on constructed decision trees. The analysis includes graphical representations of predictions and likelihood estimates for profits over different investment horizons.

--- 

### Features:
- Stock Data Retrieval: Fetches historical stock data using the yfinance library based on user input.
- Interactive Data Visualization: Plots the closing price trend for the provided stock or index over its entire history.
- Next-Day Price Prediction: Uses a RandomForest model to predict if the stock price will rise the next day based on historical data.
- Rolling Backtest: Evaluates the effectiveness of the prediction model with historical data using a rolling backtest method.
- Feature Engineering: Calculates rolling averages and trends for various time frames to enhance the prediction model.
- Profit Likelihood Analysis: Provides likelihood estimates of making a profit over different investment horizons such as 2 weeks, 1 month, or 1 quarter.

---

### Prerequisites:
Before running the Stock Price Predictor, make sure you have the following installed:

- Python (version 3.7 or above)
- yfinance library
- pandas library
- sklearn library
- matplotlib library

---

### Installation:
- Clone the repository: git clone https://github.com/your-username/Stock-Profit-Predictor.git
- Install the required dependencies: pip install -r requirements.txt

---

### Usage:
- Run the program: python stock_predictor.py
- Input the desired stock ticker symbol when prompted.
- Interact with the data plots as they appear.
- Review the predictions and profit likelihood estimates displayed.

---

### Practical Uses:
- Stock Market Analysis: Helps in gauging the short-term movements of specific stocks or market indices.
- Investment Strategy: Assists investors in making informed decisions by providing likelihood estimates for different investment horizons.
- Financial Research: A tool for scholars and researchers for empirical finance or quantitative analysis projects.
- Model Testing: A platform for testing and evaluating machine learning models in stock price prediction scenarios.

### Acknowledgments
- yfinance library - Used for fetching stock data.
- RandomForestClassifier from sklearn - Used for making stock price rise predictions.
