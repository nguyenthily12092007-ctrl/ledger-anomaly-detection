import streamlit as st
import pandas as pd

st.title("Phát hiện giao dịch bất thường")

# Đọc kết quả
df = pd.read_csv('data/anomaly_results.csv')

# Hiển thị dữ liệu
st.subheader("Danh sách giao dịch")

st.dataframe(df)

# Lọc giao dịch bất thường
anomalies = df[df['anomaly'] == -1]

st.subheader("Giao dịch bất thường")

st.dataframe(anomalies)

st.write("Số giao dịch bất thường:", len(anomalies))