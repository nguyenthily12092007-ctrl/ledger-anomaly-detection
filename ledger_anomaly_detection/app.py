import streamlit as st
import pandas as pd

st.title("Phát hiện giao dịch bất thường")

# Đọc kết quả
df = pd.read_csv('data/anomaly_results.csv')

# Hiển thị dữ liệu
st.subheader("Danh sách giao dịch ")

st.dataframe(df)

# Lọc giao dịch bất thường
anomalies = df[df['anomaly'] == -1]

# Kiểm tra nếu trong file kết quả có cột điểm số bất thường 
if 'anomaly_score' in anomalies.columns:
    # Sắp xếp các giao dịch rủi ro cao nhất (điểm số âm nhất) lên trên cùng
    anomalies = anomalies.sort_values(by='anomaly_score', ascending=True)
st.subheader("Các giao dịch bất thường")

st.dataframe(anomalies)

st.write("Số giao dịch bất thường:", len(anomalies))
