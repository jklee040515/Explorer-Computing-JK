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

st.set_page_config(
    page_title="서울 강수량 기반 지하철 이용량 분석",
    layout="wide"
)

st.title("🌧️ 서울 강수량 기반 지하철 이용량 분석 & 예측 웹앱")
st.markdown("""
## 📘 연구 배경 및 목적
대도시 서울에서는 하루 수백만 명이 지하철을 이용하며, 지하철 이용량의 변동은 **열차 배차, 혼잡 관리, 안전 인력 배치** 등 도시 교통 운영 전반에 큰 영향을 미칩니다.
일반적으로 지하철은 버스나 자가용에 비해 **날씨의 영향을 덜 받을 것이라는 가설**이 존재하지만, 이를 **실제 데이터로 검증한 연구는 많지 않습니다.**
본 프로젝트는 **2015~2025년 서울시 월별 강수량 데이터와 지하철 승·하차 인원 데이터를 결합**하여 강수량이 지하철 이용량에 미치는 영향을 데이터 기반으로 분석하고자 합니다.""")
st.write("2015~2025년 데이터를 기반으로 강수량과 지하철 승하차 인원의 관계를 분석하고 예측합니다.")

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

with st.expander("📂 프로젝트 진행 과정", expanded=False):
    st.markdown(""" 
### ① 데이터 수집
- 기상 데이터: 월별 서울 강수량 데이터 (2015~2025)
- 교통 데이터: 서울시 지하철 월별 승·하차 인원 데이터

### ② 데이터 전처리
- 다양한 날짜 형식(yyyy-mm, yyyy년 m월 등)을 `year_month`로 통일
- 월 단위 기준으로 강수량 및 승하차 인원 집계

### ③ 데이터 병합
- `year_month` 기준으로 강수량 데이터와 지하철 데이터를 병합
- 분석 가능한 단일 테이블 생성

### ④ 분석 및 모델링
- 피어슨 상관계수 계산
- 단순 선형 회귀 모델 학습 및 결정계수(R²) 산출

### ⑤ 시각화 및 예측
- 강수량 vs 지하철 이용량 산점도 및 회귀선
- 실시간 강수량 기반 지하철 이용량 예측 기능 구현
""")

rain_df = load_csv(RAIN_PATH)
sub_df = load_csv(SUBWAY_PATH)

st.subheader("📄 강수량 데이터 (미리보기)")
st.dataframe(rain_df.head())

st.subheader("📄 지하철 승하차 데이터 (미리보기)")
st.dataframe(sub_df.head())

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

st.header("3. 데이터 병합")

merged = pd.merge(rain_m, sub_m, on="year_month", how="inner")
st.dataframe(merged.head())

if merged.empty:
    st.error("⚠ 병합된 데이터가 없습니다. 날짜 형식을 확인하세요.")
    st.stop()

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

st.markdown("""
### 📊 분석 결과 해석

- **피어슨 상관계수는 약 0.1~0.2 수준**으로 나타나 강수량과 지하철 승하차 인원 간의 상관관계는 매우 약한 편입니다.
- **결정계수(R²)는 0.02 내외**로, 강수량 하나만으로는 지하철 이용량 변화를 거의 설명하지 못합니다.
👉 이는 지하철 이용량이 강수량보다는 **출퇴근 패턴, 요일, 공휴일, 계절성** 등의 구조적인 요인에 의해 더 크게 결정된다는 점을 시사합니다.
""")

st.header("5. 강수량 vs 지하철 이용량 시각화")

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(X, y, label="data")
ax.plot(X, model.predict(X_scaled), label="regression line")

ax.set_xlabel("Precipitation (mm)")
ax.set_ylabel("Subway passengers")
ax.legend()

st.pyplot(fig)

st.markdown("""
## 🧾 결론 및 시사점

- 강수량은 지하철 이용량에 **미미한 영향만을 미침**
- 지하철 운영 정책은 기상 변수보다는 **고정적인 통근·통학 패턴을 중심으로 설계**되어야 함
- 본 연구는 **기상 데이터와 도시 교통 데이터를 결합한 기초 분석 사례**로서 의미를 가짐
""")


st.header("6. 실시간 강수량 기반 지하철 이용량 예측")

def get_today_rain():
    url = "https://weather.naver.com/today/09140580"
    html = requests.get(url, timeout=5).text
    soup = BeautifulSoup(html, "html.parser")

    texts = soup.stripped_strings

    for text in texts:
        matches = re.findall(r"\d+\.?\d*", text)
        for m in matches:
            try:
                value = float(m)
                # 강수량은 비정상적으로 큰 값이 나오지 않음
                if 0 <= value <= 500:
                    return value
            except ValueError:
                continue

    # 파싱 실패 시 안전한 기본값
    return 0.0


if st.button("오늘 강수량 가져와서 예측하기"):
    today_rain = get_today_rain()
    st.success(f"오늘 강수량: **{today_rain} mm**")

    scaled_value = scaler.transform([[today_rain]])
    pred = model.predict(scaled_value)[0]

    st.info(f"📌 예상 지하철 승하차 인원: **{pred:,.0f} 명**") 

st.markdown("""
## 🔮 연구의 한계 및 향후 확장

- 본 연구는 **강수량 단일 변수**만을 사용한 단순 회귀 모델임
- 향후 연구에서는  
  - 기온, 습도, 미세먼지  
  - 요일, 공휴일, 계절성 변수 등을 추가한 **다중 회귀 분석**으로 확장 가능
- 추가 변수를 포함할 경우  
  **실제 도시 교통 수요 예측 시스템으로의 발전 가능성**이 있음
""")







