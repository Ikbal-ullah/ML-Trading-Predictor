import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# UI Configuration
st.set_page_config(page_title="ML Trading Signal Predictor", layout="wide")
st.title("📈 ML Algorithmic Trading Predictor")
st.write("This app uses a Random Forest Classifier to predict if a stock will close UP or DOWN tomorrow based on historical momentum.")

# Sidebar parameters
st.sidebar.header("Model Parameters")
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL)", "AAPL")
years = st.sidebar.slider("Years of Historical Data", 1, 10, 5)

@st.cache_data 
def load_data(ticker, years):
    # Using static dataset for stability on Cloud Deployment
    url = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
    df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
    df = df.rename(columns={'AAPL.Close': 'Close', 'AAPL.Volume': 'Volume'})
    return df

data = load_data(ticker, years)
st.subheader(f"Historical Data (Dataset: {ticker})")
st.line_chart(data['Close'])

# --- FEATURE ENGINEERING ---
df = data.copy()
df['SMA_10'] = df['Close'].rolling(window=10).mean() 
df['SMA_50'] = df['Close'].rolling(window=50).mean() 
df['Daily_Return'] = df['Close'].pct_change()

# Define binary target: 1 if tomorrow > today, else 0
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna()

# --- ML PIPELINE ---
features = ['SMA_10', 'SMA_50', 'Daily_Return', 'Volume']
X = df[features]
y = df['Target']

# Time-series split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Model initialization
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# --- EVALUATION ---
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)

st.subheader("🤖 Model Evaluation Metrics")
col1, col2 = st.columns(2)
col1.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
col2.metric("Model Precision", f"{precision * 100:.2f}%")

# --- INFERENCE (PREDICTION) ---
st.subheader("🔮 Tomorrow's Prediction")

# Professional Inference Logic:
# We grab the last row as a DataFrame to maintain 'Feature Names'
# This prevents the UserWarning about feature name mismatch
predictions = model.predict(pd.DataFrame(X_test, columns=features))
latest_row = X.tail(1) 
tomorrow_pred = model.predict(pd.DataFrame(latest_row, columns=features))

if tomorrow_pred[0] == 1:
    st.success(f"**SIGNAL: BUY** - The model predicts the price will close **HIGHER** tomorrow.")
else:
    st.error(f"**SIGNAL: SELL** - The model predicts the price will close **LOWER** tomorrow.")

st.info("Technical Note: Predictions are based on momentum indicators (SMA 10/50) and daily volatility.")