import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import joblib

# Đọc dữ liệu
df = pd.read_csv('data/financial_anomaly_data.csv')

print(df.head())

# Encode dữ liệu text
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))

# Chọn dữ liệu số
features = df.select_dtypes(include=['int64', 'float64'])
# --- CODE BỔ SUNG TRÊN NHÁNH feature-improve-model ---
from sklearn.preprocessing import StandardScaler

# Khởi tạo bộ chuẩn hóa để đưa các biến số về cùng một thang đo
scaler = StandardScaler()
# Giặt sạch dữ liệu số
features_scaled = scaler.fit_transform(features)

# Thay vì dùng 'features' gốc, bạn sửa lại các hàm phía dưới thành:
# model.fit(features_scaled)
# predictions = model.predict(features_scaled)
# df['anomaly_score'] = model.decision_function(features_scaled)
# Tạo model
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

# Train model
model.fit(features)

# Predict anomaly
predictions = model.predict(features)

# Thêm cột anomaly
df['anomaly'] = predictions

# Lưu model
joblib.dump(model, 'models/isolation_forest.pkl')

# Lưu kết quả
df.to_csv('data/anomaly_results.csv', index=False)

print('Hoàn thành phát hiện giao dịch bất thường!')
