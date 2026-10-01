import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ---------------------------------
# 페이지 설정
# ---------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.write(
    "1908년부터 2025년까지의 서울 연평균기온을 이용해 "
    "장기적인 기온 변화를 회귀선으로 살펴봅니다."
)


# ---------------------------------
# 데이터 불러오기
# ---------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 숫자로 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# ---------------------------------
# 연도별 평균기온 계산
# ---------------------------------
yearly = (
    df.dropna(subset=["연도", "평균기온"])
    .groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)


# ---------------------------------
# 분석 대상 연도 선택
# 조건:
# 1. 2025년까지
# 2. 1908년 이후
# 3. 관측일수 300일 이상
# ---------------------------------
yearly = yearly[
    (yearly["연도"] >= 1908)
    & (yearly["연도"] <= 2025)
    & (yearly["관측일수"] >= 300)
].copy()


yearly = yearly.sort_values("연도").reset_index(drop=True)


# ---------------------------------
# 회귀분석
# 독립변수: 1908년부터 지난 연수
# 예:
# 1908년 → 0
# 1909년 → 1
# 2025년 → 117
# ---------------------------------
yearly["지난연수"] = yearly["연도"] - 1908

x = yearly["지난연수"].to_numpy()
y = yearly["연평균기온"].to_numpy()


# 회귀계수
slope, intercept = np.polyfit(x, y, 1)

# 회귀선의 예측값
yearly["회귀예측기온"] = intercept + slope * yearly["지난연수"]


# 상관계수
correlation = np.corrcoef(x, y)[0, 1]


# ---------------------------------
# 예측 함수
# ---------------------------------
def predict_temperature(year):
    elapsed_years = year - 1908
    return intercept + slope * elapsed_years


# ---------------------------------
# 분석 정보
# ---------------------------------
st.subheader("📊 분석에 사용한 데이터")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("회귀에 사용한 연도 수", f"{len(yearly)}년")

with col2:
    st.metric("시작 연도", f"{int(yearly['연도'].min())}년")

with col3:
    st.metric("끝 연도", f"{int(yearly['연도'].max())}년")

with col4:
    st.metric("상관계수", f"{correlation:.3f}")


st.write(
    f"※ 2025년까지의 자료 중 **관측일수가 300일 이상인 연도**만 분석에 사용했습니다."
)


# ---------------------------------
# 회귀식 설명
# ---------------------------------
st.subheader("📐 회귀선")

st.write(
    f"회귀식: **예상 기온 = {intercept:.3f} + "
    f"({slope:.4f} × 지난 연수)**"
)

st.write(
    f"1908년을 기준으로 할 때, 연평균기온은 회귀선상에서 "
    f"1년에 약 **{slope:.4f}℃**씩 변하는 것으로 계산됩니다."
)


# ---------------------------------
# 산점도 + 회귀선
# ---------------------------------
st.subheader("🌡️ 연도별 연평균기온과 회귀선")

fig = go.Figure()


# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=yearly["관측일수"],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        )
    )
)


# 회귀선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["회귀예측기온"],
        mode="lines",
        name="회귀선",
        line=dict(width=3),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "회귀 예측값: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    hovermode="x unified",
    height=600,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------
# 기온 예측 슬라이더
# ---------------------------------
st.subheader("🔮 기온 예측기")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


predicted_temperature = predict_temperature(selected_year)


st.markdown(
    f"""
    <div style="
        background-color:#f0f7ff;
        padding:30px;
        border-radius:15px;
        text-align:center;
        margin-top:20px;
        margin-bottom:20px;
    ">
        <div style="font-size:22px; color:#555;">
            {selected_year}년 회귀선 기준 예상 연평균기온
        </div>
        <div style="font-size:55px; font-weight:bold;">
            {predicted_temperature:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------
# 선택한 연도의 위치를 보여주는 그래프
# ---------------------------------
prediction_years = np.arange(1900, 2101)
prediction_values = [
    predict_temperature(year)
    for year in prediction_years
]


fig_prediction = go.Figure()


# 회귀선 전체
fig_prediction.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_values,
        mode="lines",
        name="회귀선",
        line=dict(width=3)
    )
)


# 선택한 연도
fig_prediction.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temperature],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=14),
        hovertemplate=(
            f"<b>{selected_year}년</b><br>"
            f"예상 연평균기온: {predicted_temperature:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig_prediction.update_layout(
    xaxis_title="연도",
    yaxis_title="예상 연평균기온 (℃)",
    xaxis=dict(
        range=[1900, 2100],
        tickmode="linear",
        dtick=10
    ),
    height=450,
    hovermode="closest"
)


st.plotly_chart(fig_prediction, use_container_width=True)


# ---------------------------------
# 주의사항
# ---------------------------------
st.info(
    "이 값은 과거 연평균기온과 연도 사이의 선형 관계를 단순한 "
    "회귀식으로 연장한 값입니다. 실제 미래 기온을 정확하게 예측하는 "
    "기상예보 값은 아닙니다."
)
