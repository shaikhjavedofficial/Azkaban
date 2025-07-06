import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from hmmlearn.hmm import GaussianHMM
import ta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import datetime

# --- Parameters ---
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start_date = '2021-01-01'
end_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
window = 5  # For rolling stats
print("welcome to FinSage, a stock prediction model using a stacked ensemble of machine learning algorithms.")
# --- Data Download and Feature Engineering ---
all_data = []
for symbol in symbols:
    df = yf.download(symbol, start=start_date, end=end_date)
    df['Return'] = df['Close'].pct_change()
    df['Direction'] = (df['Return'] > 0).astype(int)
    df['Rolling_Mean_5'] = df['Close'].rolling(window=window).mean()
    df['Rolling_Std_5'] = df['Close'].rolling(window=window).std()
    # Add RSI and MACD
    close_series = df['Close'].squeeze()  # Ensure it's a Series, not DataFrame
    df['RSI'] = ta.momentum.RSIIndicator(close_series, window=14).rsi()
    macd = ta.trend.MACD(close_series)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df.dropna(inplace=True)
    df['Symbol'] = symbol
    all_data.append(df)
    

full_df = pd.concat(all_data)
full_df.to_csv('training_data.csv', index=False)
full_df.reset_index(inplace=True)

# --- Regime Detection (HMM) ---
def detect_regimes(returns, n_states=3):
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
    returns = returns.values.reshape(-1, 1)
    model.fit(returns)
    hidden_states = model.predict(returns)
    return hidden_states

regime_labels = {0: 'Regime_0', 1: 'Regime_1', 2: 'Regime_2'}

# Apply regime detection per stock
full_df['Regime'] = None
for symbol in symbols:
    idx = full_df['Symbol'] == symbol
    regimes = detect_regimes(full_df.loc[idx, 'Return'])
    # Assign regime names based on mean return in each regime
    regime_means = pd.Series(regimes).groupby(regimes).mean()
    regime_order = np.argsort(regime_means)
    regime_map = {regime_order[2]: 'Bull', regime_order[0]: 'Bear', regime_order[1]: 'Sideways'}
    full_df.loc[idx, 'Regime'] = [regime_map[r] for r in regimes]

# --- Movement Prediction (Stacked Ensemble) ---
# Add new features
features = ['Rolling_Mean_5', 'Rolling_Std_5', 'RSI', 'MACD', 'MACD_Signal']
X = full_df[features]
y = full_df['Direction']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=full_df['Symbol']
)

base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
]
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(),
    passthrough=True,
    n_jobs=-1
)
stacking_clf.fit(X_train, y_train)
y_pred = stacking_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- Output Results ---
output_df = X_test.copy()
output_df['Actual_Direction'] = y_test
output_df['Predicted_Direction'] = y_pred
output_df['Symbol'] = full_df.loc[output_df.index, 'Symbol']
output_df['Date'] = full_df.loc[output_df.index, 'Date']
output_df['Regime'] = full_df.loc[output_df.index, 'Regime']
output_df = output_df[['Date', 'Symbol', 'Rolling_Mean_5', 'Rolling_Std_5', 'Regime', 'Actual_Direction', 'Predicted_Direction']]
output_df.to_csv('stock_predictions_with_regimes.csv', index=False)

print(f"Test Accuracy: {accuracy:.2%}")
print("Results saved to 'stock_predictions_with_regimes.csv'")

# For each symbol, get its regime, a description, and the overall accuracy

def regime_description(regime):
    if regime == 'Bull':
        return ("The model has detected a Bull regime, characterized by high returns and a strong upward trend. "
                "This suggests positive momentum and favorable conditions for investment. "
                "Based on historical patterns, you may consider investing, but always assess your risk tolerance and market conditions.")
    elif regime == 'Bear':
        return ("The model has detected a Bear regime, characterized by low returns and a downward trend. "
                "This indicates negative momentum and increased risk. "
                "It is generally advisable to avoid new investments or consider defensive strategies during this period.")
    elif regime == 'Sideways':
        return ("The model has detected a Sideways regime, where returns are stable or fluctuating without a clear trend. "
                "Market direction is uncertain, and significant gains are less likely. "
                "You may choose to wait for a clearer trend before making investment decisions.")
    else:
        return "Unknown regime. Please review the data and model outputs for more information."

summary_rows = []
for symbol in symbols:
    # Filter test set for this symbol
    symbol_idx = output_df['Symbol'] == symbol
    actual = output_df.loc[symbol_idx, 'Actual_Direction']
    predicted = output_df.loc[symbol_idx, 'Predicted_Direction']
    # Calculate accuracy for this symbol
    if len(actual) > 0:
        symbol_accuracy = accuracy_score(actual, predicted)
    else:
        symbol_accuracy = float('nan')
    # Get most recent regime
    symbol_rows = full_df[full_df['Symbol'] == symbol]
    regime = symbol_rows['Regime'].iloc[-1]
    desc = regime_description(regime)
    summary_rows.append({
        'Symbol': symbol,
        'Regime': regime,
        'Description': desc,
        'Accuracy': f"{symbol_accuracy:.2%}"
    })

summary_df = pd.DataFrame(summary_rows, columns=['Symbol', 'Regime', 'Description', 'Accuracy'])
summary_df.to_csv('stock_regime_summary.csv', index=False)

print("Summary saved to 'stock_regime_summary.csv'")
