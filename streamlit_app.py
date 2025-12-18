import streamlit as st
import pandas as pd
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

plt.rcParams["axes.unicode_minus"] = False

st.title("🌧️ 서울 강수량과 지하철 승하차 인원 상관관계 분석")
st.write("2015~2025년 월별 데이터를 활용한 데이터 기반 분석")

# ===============================
# 연구 배경
# ===============================
st.markdown("""
## 📘 연구 배경 및 목적
서울의 지하철 이용량은 도시 교통 운영의 핵심 지표이다.  
본 연구는 **강수량이 지하철 이용량에 유의미한 영향을 미치는지**를
실제 데이터를 통해 검증하는 것을 목적으로 한다.
""")

# ===============================
# 데이터 불러오기
# ===============================
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

# ===============================
# 날짜 파싱 (🔥 핵심 함수)
# ===============================
def parse_month(df):
    col = df.iloc[:, 0].astype(str)
    extracted = col.str.extract(r"(20\d{2})\D*([01]?\d)").dropna()

    if extracted.empty:
        return pd.DataFrame(columns=["year_month"])

    extracted[1] = extracted[1].astype(int)

    df = df.loc[extracted.index].copy()
    df["year_month"] = pd.PeriodIndex(
        extracted[0].astype(str) + "-" + extracted[1].astype(str),
        freq="M"
    )
    return df

# ===============================
# 강수량 전처리
# ===============================
def prep_rain(df):
    df = parse_month(df)

    # year_month 제외한 나머지 컬럼 후보
    candidate_cols = [c for c in df.columns if c != "year_month"]

    # 각 컬럼에서 숫자 추출 시도
    best_col = None
    max_valid = 0

    for c in candidate_cols:
        # 문자열에서 숫자만 추출
        temp = (
            df[c]
            .astype(str)
            .str.extract(r"([-+]?\d*\.?\d+)")
            .astype(float)
        )

        valid_count = temp.notna().sum()

        # 가장 숫자가 많이 살아남은 컬럼을 강수량으로 선택
        if valid_count > max_valid:
            max_valid = valid_count
            best_col = c
            df[c] = temp[0]

    if best_col is None or max_valid < 2:
        st.error("⚠ 강수량 데이터를 식별할 수 없습니다. CSV 형식을 확인하세요.")
        st.stop()

    return df[["year_month", best_col]].rename(columns={best_col: "precip_mm"})

# ===============================
# 지하철 전처리
# ===============================
def prep_subway(df):
    df = parse_month(df)
    num_cols = df.select_dtypes(include="number").columns
    df["passengers"] = df[num_cols].sum(axis=1)
    return df.groupby("year_month", as_index=False)[["passengers"]].sum()

# ===============================
# 전처리 실행
# ===============================
rain_m = prep_rain(rain_df)
sub_m = prep_subway(sub_df)

# ===============================
# 병합
# ===============================
merged = pd.merge(rain_m, sub_m, on="year_month", how="inner")
merged = merged.dropna(subset=["precip_mm", "passengers"])

if len(merged) < 2:
    st.error("⚠ 분석에 필요한 데이터가 충분하지 않습니다.")
    st.stop()

st.dataframe(merged.head())

# ===============================
# 상관 & 회귀 분석
# ===============================
st.header("📊 상관 분석 결과")

corr = merged["precip_mm"].corr(merged["passengers"])
st.write(f"**피어슨 상관계수:** {corr:.4f}")

X = merged[["precip_mm"]]
y = merged["passengers"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

r2 = model.score(X_scaled, y)
st.write(f"**결정계수 R²:** {r2:.4f}")

st.markdown("""
- 상관계수는 **매우 약한 양의 상관**
- R² 값이 매우 낮아 강수량의 설명력은 제한적
""")

# ===============================
# 시각화
# ===============================
st.header("📈 시각화")

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(merged["precip_mm"], merged["passengers"], alpha=0.6)
ax.plot(merged["precip_mm"], model.predict(X_scaled), linestyle="--")

ax.set_xlabel("Precipitation (mm)")
ax.set_ylabel("Subway passengers")
st.pyplot(fig)

# ===============================
# 실시간 예측
# ===============================
st.header("🔮 실시간 강수량 기반 예측")

def get_today_rain():
    url = "https://weather.naver.com/today/09140580"
    soup = BeautifulSoup(requests.get(url, timeout=5).text, "html.parser")
    for text in soup.stripped_strings:
        for m in re.findall(r"\d+\.?\d*", text):
            v = float(m)
            if 0 <= v <= 500:
                return v
    return 0.0

if st.button("오늘 강수량으로 예측"):
    today_rain = get_today_rain()
    pred = model.predict(scaler.transform([[today_rain]]))[0]
    st.success(f"강수량: {today_rain} mm")
    st.info(f"예상 지하철 이용량: {pred:,.0f} 명")

st.caption("※ 실시간 강수량은 외부 크롤링 결과이며 0으로 표시될 수 있음")

