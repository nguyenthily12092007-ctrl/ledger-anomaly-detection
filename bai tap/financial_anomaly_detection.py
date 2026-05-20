import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def generate_synthetic_financial_data(n_samples=500, random_state=42):
    """Tạo dữ liệu giao dịch tài chính giả lập với một số điểm dị thường."""
    rng = np.random.RandomState(random_state)
    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="D")
    amounts = rng.normal(loc=1000, scale=250, size=n_samples)
    amounts = np.clip(amounts, 50, None)

    # Thêm một số điểm dị thường (giao dịch cao bất thường)
    anomalies = rng.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    amounts[anomalies] *= rng.uniform(3, 10, size=len(anomalies))

    data = pd.DataFrame({
        "date": dates,
        "amount": amounts,
        "customer_id": rng.randint(1000, 1020, size=n_samples),
        "transaction_type": rng.choice(["payment", "refund", "transfer"], size=n_samples, p=[0.7, 0.15, 0.15]),
    })
    data["is_known_anomaly"] = False
    data.loc[anomalies, "is_known_anomaly"] = True
    return data


def detect_anomalies_zscore(data, column="amount", threshold=3.0):
    """Phát hiện dị thường bằng phương pháp Z-score trên cột dữ liệu số."""
    values = data[column].astype(float)
    mean = values.mean()
    std = values.std(ddof=0)
    z_scores = (values - mean) / std
    data = data.copy()
    data["z_score"] = z_scores
    data["anomaly_zscore"] = np.abs(z_scores) > threshold
    return data


def detect_anomalies_isolation_forest(data, columns=None, contamination=0.05, random_state=42):
    """Phát hiện dị thường bằng Isolation Forest đa biến."""
    if columns is None:
        columns = ["amount"]
    model = IsolationForest(contamination=contamination, random_state=random_state)
    data = data.copy()
    X = data[columns].astype(float)
    model.fit(X)
    data["anomaly_iforest"] = model.predict(X) == -1
    return data


def summarize_anomalies(data):
    """Tóm tắt kết quả phát hiện dị thường."""
    return {
        "total_records": len(data),
        "zscore_anomalies": int(data["anomaly_zscore"].sum()),
        "iforest_anomalies": int(data["anomaly_iforest"].sum()),
        "known_anomalies": int(data.get("is_known_anomaly", pd.Series(dtype=bool)).sum()),
    }


def main():
    data = generate_synthetic_financial_data(n_samples=500)
    data = detect_anomalies_zscore(data, column="amount", threshold=3.0)
    data = detect_anomalies_isolation_forest(data, columns=["amount"], contamination=0.05)

    summary = summarize_anomalies(data)
    print("=== Tóm tắt phát hiện dị thường ===")
    print(f"Tổng số bản ghi: {summary['total_records']}")
    print(f"Dị thường Z-score: {summary['zscore_anomalies']}")
    print(f"Dị thường Isolation Forest: {summary['iforest_anomalies']}")

    output_file = "financial_anomaly_data.csv"
    data.to_csv(output_file, index=False)
    print(f"Đã lưu dữ liệu và nhãn dị thường vào: {output_file}")

    print("\nMột số bản ghi dị thường (theo Isolation Forest):")
    print(data.loc[data["anomaly_iforest"], ["date", "amount", "transaction_type", "anomaly_zscore", "anomaly_iforest"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
