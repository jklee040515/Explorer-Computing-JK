import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path

# ===============================
# Streamlit 설정
# ===============================
st.set_page_config(
    page_title="서울 강수량 기반 지하철 이용량 분석",
    layout="wide"
)

# ⚠ Streamlit Cloud에는 Malgun Gothic 폰트 없음 → 제거
plt.rcParams['axes.unicode_minus'] = False

st.title("🌧️ 서울 강수량 기반 지하철 이용량 분석 & 예측 웹앱")
st.write("2015~2025년 데이터를 기반으로 강수량과 지하철 승하차 인원의 관계를 분석하고 예측합니다.")

# ===============================
# 1. 데이터 불러오기
# ===============================
st.header("1. 데이터 불러오기")

# 🔹 Streamlit Cloud 대응 경로 설정
BASE_DIR = Path(__file__).resolve().parent

RAIN_PATH = BASE_DIR / "2015~2025 월별 서울시 강수량.csv"
SUBWAY_PATH = BASE_DIR / "2015~2025 월별 서울시 지하철 승하차 인원.csv"


def load_csv(path):
    """CSV 파일을 UTF-8 또는 CP949로 자동 인코딩하여 불러옴"""
    try:
        return pd.read_csv(path)
    except:
        return pd.read_csv(path, encoding="cp949")


rain_df = load_csv(RAIN_PATH)
sub_df = load_csv(SUBWAY_PATH)

st.subheader("📄 강수량 데이터 (미리보기)")
st.dataframe(rain_df.head())

st.subheader("📄 지하철 승하차 데이터 (미리보기)")
st.dataframe(sub_df.head())

# ===============================
# 2. year_month 파싱 공통 함수
# ===============================
def parse_month(df):
    df = df.copy()
    col = df.iloc[:, 0].astype(str)

    extracted = col.str.extract(r"(20[0-9]{2})[^0-9]*([0-9]{1,2})")
    extracted = extracted.dropna()

    if extracted.empty:
        tmp = pd.to_datetime(col, format="%b-%y", errors="coerce")
        df["year_month"] = tmp.dt.to_period("M").dt.to_timestamp()
        return df.dropna(subset=["year_month"])

    extracted[1] = extracted[1].astype(int).apply(lambda x: f"{x:02d}")
    df = df.loc[extracted.index]
    df["year_month"] = pd.to_datetime(extracted[0] + "-" + extracted[1])

    return df.dropna(subset=["year_month"])

# ===============================
# 3. 강수량 데이터 전처리
# ===============================
def prep_rain(df):
    df = parse_month(df)

    rain_col = None
    for c in df.columns:
        if "강수" in c or "rain" in c or "mm" in c:
            rain_col = c

    if rain_col is None:
        rain_col = df.select_dtypes(include="number").columns[0]

    df[rain_col] = pd.to_numeric(df[rain_col], errors="coerce")
    df = df[["year_month", rain_col]]

    return df.rename(columns={rain_col: "precip_mm"})

# ===============================
# 4. 지하철 데이터 전처리
# ===============================
def prep_subway(df):
    df = parse_month(df)
    num_cols = df.select_dtypes(include="number").columns

    if len(num_cols) == 0:
        raise ValueError("숫자 승하차 인원 컬럼을 찾을 수 없습니다.")

    df["passengers"] = df[num_cols].sum(axis=1)
    df = df.groupby("year_month", as_index=False).sum()

    return df[["year_month", "passengers"]]


rain_m = prep_rain(rain_df)
sub_m = prep_subway(sub_df)

st.header("2. 전처리된 데이터 미리보기")

st.subheader("🌧 월별 강수량")
st.dataframe(rain_m.head())

st.subheader("🚇 월별 지하철 승하차 인원")
st.dataframe(sub_m.head())

# ===============================
# 5. 데이터 병합
# ===============================
st.header("3. 데이터 병합")

merged = pd.merge(rain_m, sub_m, on="year_month", how="inner")
st.dataframe(merged.head())

if merged.empty:
    st.error("⚠ 병합된 데이터가 없습니다. 날짜 형식을 확인하세요.")
    st.stop()

# ===============================
# 6. 상관 분석 & 회귀 분석
# ===============================
st.header("4. 상관 분석 및 회귀 분석")

corr = merged["precip_mm"].corr(merged["passengers"])
st.write(f"📌 **피어슨 상관계수:** {corr:.4f}")

X = merged[["precip_mm"]]
y = merged["passengers"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

r2 = model.score(X_scaled, y)
st.write(f"📌 **결정계수 R²:** {r2:.4f}")

# ===============================
# 7. 시각화
# ===============================
st.header("5. 강수량 vs 지하철 이용량 시각화")

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(X, y, label="실제 데이터")
ax.plot(X, model.predict(X_scaled), label="회귀선")

ax.set_xlabel("강수량 (mm)")
ax.set_ylabel("승하차 인원")
ax.legend()

st.pyplot(fig)

# ===============================
# 8. 실시간 강수량 크롤링 + 예측
# ===============================
st.header("6. 실시간 강수량 기반 지하철 이용량 예측")

def get_today_rain():
    url = "https://weather.naver.com/today/09140580"
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(text=re.compile(r"[0-9.]+")):
        nums = re.findall(r"[0-9.]+", tag)
        if nums:
            return float(nums[0])

    return 0.0


if st.button("오늘 강수량 가져와서 예측하기"):
    today_rain = get_today_rain()
    st.success(f"오늘 강수량: **{today_rain} mm**")

    scaled_value = scaler.transform([[today_rain]])
    pred = model.predict(scaled_value)[0]

    st.info(f"📌 예상 지하철 승하차 인원: **{pred:,.0f} 명**")

