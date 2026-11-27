import re
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

app = Flask(__name__)

CORS(
    app,
    origins=[
        "https://atrisanariim2628-maker.github.io",
        re.compile(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"),
    ],
)

DEFAULT_TICKERS = "RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS,ITC.NS,TATAMOTORS.NS"
DEFAULT_AMOUNT = 100000.0
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-01-01"


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "endpoint": "/api"})


@app.route("/api", methods=["GET"])
def optimize():
    raw_tickers = request.args.get("tickers", DEFAULT_TICKERS)
    tickers = [t.strip() for t in raw_tickers.split(",") if t.strip()]
    if len(tickers) < 2:
        return jsonify({"error": "Provide at least 2 comma-separated tickers."}), 400

    try:
        investment_amount = float(request.args.get("amount", DEFAULT_AMOUNT))
        if investment_amount <= 0:
            raise ValueError("amount must be positive")
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid 'amount' parameter; must be a positive number."}), 400

    start = request.args.get("start", DEFAULT_START).strip() or DEFAULT_START
    end = request.args.get("end", DEFAULT_END).strip() or DEFAULT_END

    try:
        # Download tickers individually to avoid batch rate-limit failures on shared IPs
        series_list = []
        failed = []
        for ticker in tickers:
            s = None
            for attempt in range(3):  # retry up to 3 times per ticker
                try:
                    import time as _time
                    if attempt > 0:
                        _time.sleep(2 ** attempt)  # 2s, 4s back-off
                    raw = yf.download(
                        ticker, start=start, end=end,
                        progress=False, threads=False, auto_adjust=True
                    )
                    if raw.empty:
                        continue
                    close_col = "Close" if "Close" in raw.columns else raw.columns[0]
                    series = raw[close_col].dropna()
                    series.name = ticker
                    s = series
                    break
                except Exception:
                    continue
            if s is not None and not s.empty:
                series_list.append(s)
            else:
                failed.append(ticker)

        if len(series_list) < 2:
            msg = "Not enough valid ticker data to optimize portfolio."
            if failed:
                msg += f" Failed/delisted tickers: {', '.join(failed)}."
            return jsonify({"error": msg}), 400

        df = pd.concat(series_list, axis=1).dropna(axis=0, how="any")
        if df.empty or len(df.columns) < 2:
            return jsonify({"error": "Not enough overlapping price data across tickers for the given date range."}), 400

        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)
        ef = EfficientFrontier(mu, S)
        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        expected_return, volatility, sharpe = ef.portfolio_performance()

        weights_formatted = {
            str(ticker): round(float(weight) * 100, 2)
            for ticker, weight in cleaned_weights.items()
            if weight > 0
        }

        latest_prices = get_latest_prices(df)
        da = DiscreteAllocation(
            cleaned_weights,
            latest_prices,
            total_portfolio_value=investment_amount,
        )
        allocation, leftover = da.lp_portfolio()

        return jsonify({
            "weights": weights_formatted,
            "allocation": {str(k): int(v) for k, v in allocation.items()},
            "leftover_cash": round(float(leftover), 2),
            "performance": {
                "expected_annual_return": round(float(expected_return) * 100, 2),
                "annual_volatility": round(float(volatility) * 100, 2),
                "sharpe_ratio": round(float(sharpe), 4),
            },
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
