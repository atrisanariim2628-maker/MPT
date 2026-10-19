from flask import Flask, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

app = Flask(__name__)

@app.route('/api', methods=['GET'])
def optimize():
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS", "TATAMOTORS.NS"]
    investment_amount = 100000
    
    # 1. Fetch data
    data = yf.download(tickers, start="2023-01-01", end="2026-01-01")['Close']
    data = data.dropna(axis=1, how='any')
    
    if data.empty or len(data.columns) < 2:
        return jsonify({"error": "Not enough data"}), 400

    # 2. Optimize
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    ef = EfficientFrontier(mu, S)
    cleaned_weights = ef.clean_weights()
    
    # Format weights for JSON output
    weights_formatted = {ticker: round(weight * 100, 2) for ticker, weight in cleaned_weights.items() if weight > 0}
    
    # 3. Allocation
    latest_prices = get_latest_prices(data)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=investment_amount)
    allocation, leftover = da.lp_portfolio()
    
    return jsonify({
        "weights": weights_formatted,
        "allocation": allocation,
        "leftover_cash": round(leftover, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)