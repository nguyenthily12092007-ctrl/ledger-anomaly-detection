import os
import threading
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from financial_anomaly_detection import (
    detect_anomalies_isolation_forest,
    detect_anomalies_zscore,
    generate_synthetic_financial_data,
    summarize_anomalies,
)

app = Flask(__name__)

TRANSACTION_TYPE_MAP = {"payment": 0, "refund": 1, "transfer": 2}
DATA_FILE = "financial_anomaly_data.csv"


def load_financial_dataset():
    if os.path.exists(DATA_FILE):
        data = pd.read_csv(DATA_FILE, parse_dates=["date"])
    else:
        data = generate_synthetic_financial_data(n_samples=500)
    data = data.copy()
    data["transaction_type_code"] = data["transaction_type"].map(TRANSACTION_TYPE_MAP).fillna(-1).astype(int)
    return data


def prepare_input_row(date_text, amount, customer_id, transaction_type):
    try:
        date = pd.to_datetime(date_text)
    except Exception:
        date = pd.Timestamp(datetime.now())

    row = {
        "date": date,
        "amount": float(amount),
        "customer_id": int(customer_id),
        "transaction_type": transaction_type,
        "transaction_type_code": TRANSACTION_TYPE_MAP.get(transaction_type, -1),
        "is_known_anomaly": False,
    }
    return pd.DataFrame([row])


@app.route("/", methods=["GET", "POST"])
def index():
    data = load_financial_dataset()
    result = None
    input_values = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "amount": 1000,
        "customer_id": 1001,
        "transaction_type": "payment",
        "contamination": 0.05,
    }

    if request.method == "POST":
        date = request.form.get("date", input_values["date"])
        amount = request.form.get("amount", input_values["amount"])
        customer_id = request.form.get("customer_id", input_values["customer_id"])
        transaction_type = request.form.get("transaction_type", input_values["transaction_type"])
        contamination = float(request.form.get("contamination", input_values["contamination"]))

        input_values = {
            "date": date,
            "amount": amount,
            "customer_id": customer_id,
            "transaction_type": transaction_type,
            "contamination": contamination,
        }

        input_row = prepare_input_row(date, amount, customer_id, transaction_type)
        analysis_data = pd.concat([data, input_row], ignore_index=True)

        analysis_data = detect_anomalies_zscore(analysis_data, column="amount", threshold=3.0)
        analysis_data = detect_anomalies_isolation_forest(
            analysis_data,
            columns=["amount", "transaction_type_code"],
            contamination=contamination,
        )

        input_index = len(analysis_data) - 1
        is_iforest_anomaly = bool(analysis_data.loc[input_index, "anomaly_iforest"])
        is_zscore_anomaly = bool(analysis_data.loc[input_index, "anomaly_zscore"])

        result = {
            "input_record": analysis_data.loc[input_index, ["date", "amount", "customer_id", "transaction_type"]].to_dict(),
            "is_iforest_anomaly": is_iforest_anomaly,
            "is_zscore_anomaly": is_zscore_anomaly,
            "iforest_score": int(analysis_data["anomaly_iforest"].sum()),
            "zscore_score": int(analysis_data["anomaly_zscore"].sum()),
            "total_records": len(analysis_data),
            "top_iforest_anomalies": analysis_data.loc[
                analysis_data["anomaly_iforest"],
                ["date", "amount", "customer_id", "transaction_type"],
            ].tail(10).to_dict(orient="records"),
            "top_zscore_anomalies": analysis_data.loc[
                analysis_data["anomaly_zscore"],
                ["date", "amount", "customer_id", "transaction_type", "z_score"],
            ].sort_values("z_score", ascending=False).head(10).to_dict(orient="records"),
        }

    preview = data.head(10).to_dict(orient="records")
    transaction_types = list(TRANSACTION_TYPE_MAP.keys())

    return render_template(
        "index.html",
        summary=summary,
        result=result,
        preview=preview,
        input_values=input_values,
        transaction_types=transaction_types,
    )


if __name__ == "__main__":
    url = "http://127.0.0.1:8501"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(debug=True, port=8501)
