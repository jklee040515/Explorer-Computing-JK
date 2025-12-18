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
    page_title="서울 강수량과 지하철 이용량 분석",
    layout="wide"
)

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

st.title("🌧️ 서울 강수량과 지하철 승하차 인원 상관관계 분석 웹앱")
st.write("2015~2025년 데이터를 기반으로 강수량과 서울시 지하철 이용량의 관계를 분석하고 예측합니다.")

# ===============================
# 📘 연구 배경 및 목적 (보고서 반영)
# ===============================
st.markdown("""
## 📘 연구 배경 및 목적

대도시 서울에서는 하루 수백만 명이 지하철을 이용하며, 지하철 이용량의 변화는 **열차 배차, 혼잡 관리, 안전 인력 배치** 등 도시 교통 운영 전반에 큰 영향을 미칩니다.
일반적으로 지하철은 버스나 자가용과 달리 **날씨의 영향을 상대적으로 덜 받을 것이라는 가설**이 존재하지만, 이를 **실제 데이터로 검증한 연구는 충분하지 않습니다.**
본 프로젝트는 **2015~2025년 서울시 월별 강수량 데이터와 지하철 승·하차 인원 데이터를 결합**하여 강수량이 지하철 이용량에 미치는 영향을 데이터 기반으로 분석하는 것을 목표로 합니다.""")

# ===============================
# 1. 데이터 불러오기
# ===============================
st.header("1. 데이터 불러오기")

BASE_DIR = Path(__file__).resolve().parent

RAIN_PATH = BASE_DIR / "2015~2025 월별 서울시 강수량.csv"
SUBWAY_PATH = BASE_DIR / "2015~2025 월별 서울시 지하철 승하차 인원.csv"


def load_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.read_csv(path, encoding="cp949")


rain_df = load_csv(RAIN_PATH)
sub_df = load_csv(SUBWAY_PATH)

st.subheader("📄 강수량 데이터 미리보기")
st.dataframe(rain_df.head())

st.subheader("📄 지하철 승하차 데이터 미리보기")
st.dataframe(sub_df.head())

# ===============================
# 프로젝트 진행 과정 (보고서 내용)
# ===============================
with st.expander("📂 프로젝트 진행 과정"):
    st.markdown("""
### ① 데이터 수집
- 월별 서울 강수량 데이터
- 서울시 지하철 월별 승·하차 인원 데이터

### ② 데이터 전처리
- 다양한 날짜 형식을 `year_month`로 통일
- 월 단위 기준으로 데이터 집계

### ③ 데이터 병합
- `year_month` 기준으로 두 데이터 통합

### ④ 분석 및 모델링
- 피어슨 상관계수 계산
- 단순 선형 회귀 모델 학습

### ⑤ 시각화 및 예측
- 산점도 및 회귀선 시각화
- 실시간 강수량 기반 예측 기능 구현
""")

# ===============================
# 2. 날짜 파싱 함수
# ===============================
def parse_month(df):
    df = df.copy()
    col = df.iloc[:, 0].astype(str)

    extracted = col.str.extract(r"(20[0-9]{2})[^0-9]*([0-9]{1,2})").dropna()

    if extracted.empty:
        tmp = pd.to_datetime(col, errors="coerce")
        df["year_month"] = tmp.dt.to_period("M").dt.to_timestamp()
        return df.dropna(subset=["year_month"])

    extracted[1] = extracted[1].astype(int).apply(lambda x: f"{x:02d}")
    df = df.loc[extracted.index]
    df["year_month"] = pd.to_datetime(extracted[0] + "-" + extracted[1])

    return df


# ===============================
# 3. 강수량 전처리
# ===============================
def prep_rain(df):
    df = parse_month(df)
    rain_col = next((c for c in df.columns if "강수" in c or "rain" in c or "mm" in c), None)
    if rain_col is None:
        rain_col = df.select_dtypes(include="number").columns[0]

    df[rain_col] = pd.to_numeric(df[rain_col], errors="coerce")
    return df[["year_month", rain_col]].rename(columns={rain_col: "precip_mm"})


# ===============================
# 4. 지하철 전처리
# ===============================
def prep_subway(df):
    df = parse_month(df)
    num_cols = df.select_dtypes(include="number").columns
    df["passengers"] = df[num_cols].sum(axis=1)
    return df.groupby("year_month", as_index=False)[["passengers"]].sum()


rain_m = prep_rain(rain_df)
sub_m = prep_subway(sub_df)

# ===============================
# 5. 데이터 병합
# ===============================
st.header("2. 데이터 병합 및 전처리 결과")

merged = pd.merge(rain_m, sub_m, on="year_month", how="inner")

merged = merged.dropna(subset=["precip_mm", "passengers"])

# 데이터 수 체크
if len(merged) < 2:
    st.error("⚠ 분석에 필요한 데이터가 충분하지 않습니다.")
    st.stop()
st.dataframe(merged.head())

# ===============================
# 6. 상관 분석 & 회귀 분석
# ===============================
st.header("3. 상관 분석 및 회귀 분석")

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
- 상관계수는 **0.1 수준의 매우 약한 양의 상관관계**
- R² 값은 **0.02 내외**로 설명력이 매우 낮음  
➡ 지하철 이용량은 강수량보다는 **출퇴근, 요일, 계절성**에 더 크게 좌우됨
""")

# ===============================
# 7. 시각화
# ===============================
st.header("4. 강수량과 지하철 이용량 시각화")

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(X, y, label="Data")
ax.plot(X, model.predict(X_scaled), label="Regression line")

ax.set_xlabel("Precipitation (mm)")
ax.set_ylabel("Subway passengers")
ax.legend()

st.pyplot(fig)

# ===============================
# 8. 결론 및 시사점
# ===============================
st.markdown("""
## 🧾 결론 및 시사점

- 강수량은 지하철 이용량에 **미미한 영향**
- 교통 운영 정책은 기상 변수보다 **고정적 통근 패턴 중심** 설계 필요
- 기상 데이터와 도시 교통 데이터를 결합한 **기초 분석 사례**
""")

# ===============================
# 9. 실시간 강수량 기반 예측
# ===============================
st.header("5. 실시간 강수량 기반 지하철 이용량 예측")

def get_today_rain():
    url = "https://weather.naver.com/today/09140580"
    html = requests.get(url, timeout=5).text
    soup = BeautifulSoup(html, "html.parser")

    for text in soup.stripped_strings:
        for m in re.findall(r"\d+\.?\d*", text):
            try:
                v = float(m)
                if 0 <= v <= 500:
                    return v
            except:
                pass
    return 0.0


if st.button("오늘 강수량으로 예측하기"):
    today_rain = get_today_rain()
    pred = model.predict(scaler.transform([[today_rain]]))[0]

    st.success(f"오늘 강수량: **{today_rain} mm**")
    st.info(f"📌 예상 지하철 승하차 인원: **{pred:,.0f} 명**")

st.caption("※ 실시간 강수량은 외부 웹 크롤링 결과로, 네트워크 환경에 따라 0으로 표시될 수 있습니다.")




