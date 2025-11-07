import streamlit as st
import pandas as pd

def show_intro():
    st.title("👨‍🔬 나의 소개 페이지")
    st.header("자기소개")
    st.markdown("""
    안녕하세요! 저는 **이진규**입니다.  
    현재 대학에서 **산림과학부**에 재학 중이며, **화학**과 **프로그래밍**에 관심이 많습니다.  
    실험에서 얻은 데이터를 코딩으로 분석하거나, 화학 반응을 시뮬레이션하는 모델을 만들어 보고 싶습니다.
    """)

    st.header("🎓 학력 및 관심 분야")
    st.markdown("""
    - **소속:** 서울대학교 산림과학부  
    - **관심분야:**  
      - 🔬 분자 구조 및 화학 반응 모델링  
      - 💻 데이터 시각화 및 시뮬레이션  
      - 🌱 환경 화학, 지속 가능한 소재 개발
    """)

    st.header("🎧 취미와 여가활동")
    st.markdown("""
    - 🎵 음악 감상 - POP, 클래식, 랩, 밴드음악 등 다양한 음악을 즐깁니다.
    - 🌍 여행 — 새로운 도시의 박물관과 자연을 탐험하는 것을 좋아합니다.  
    - 📸 사진 촬영 — 실험 장면이나 여행지의 풍경을 담는 걸 즐깁니다.
    """)

    st.header("🚀 앞으로의 목표")
    st.markdown("""
    1. **화학 반응 모델링 프로그램 제작**  
       → Python으로 화학 반응 과정을 시각화하고, 분자 구조를 예측하는 시뮬레이터를 만들어 보고 싶습니다.
    2. **Streamlit 활용 프로젝트 포트폴리오 제작**  
       → 데이터 분석 능력 강화하여 이를 시각적 웹 앱 형태로 표현하고자 합니다.
    """)

    st.header("💖 좋아하는 것")
    st.write("저는 음악과 여행을 좋아하며, 새로운 지식을 배우는 걸 즐깁니다.")
    st.markdown('가장 자주 방문하는 사이트는 [YouTube 공식 홈페이지](https://youtube.com) 입니다.')
    st.write("---")
    st.caption("Streamlit으로 만든 자기소개 예제")

def show_timetable():
    st.title("📚 나의 수업 시간표")

    times = [
        "0교시(08:00~08:50)",
        "1교시(09:00~09:50)",
        "2교시(10:00~10:50)",
        "3교시(11:00~11:50)",
        "4교시(12:00~12:50)",
        "5교시(13:00~13:50)",
        "6교시(14:00~14:50)",
        "7교시(15:00~15:50)",
        "8교시(16:00~16:50)",
    ]

    data = {
        "시간": times,
        "월": ["", "", "", "인체생물학<br>(500-L307)", "", "", "대중예술의 이해<br>(43-1-101)", "", ""],
        "화": ["", "", "생명의료윤리<br>(6-103)", "처음 배우는 서양사<br>(14-208)", "", "", "", "", ""],
        "수": ["", "", "", "인체생물학<br>(500-L307)", "", "", "대중예술의 이해<br>(43-1-101)", "", ""],
        "목": ["", "", "생명의료윤리<br>(6-103)", "처음 배우는 서양사<br>(14-208)", "", "", 
               "제지화학 및 실험<br>(200-1026)", "제지화학 및 실험<br>(200-1026)", "제지화학 및 실험<br>(200-1026)"],
        "금": ["", "", "컴퓨팅 탐색: 실생활에서 활용하기<br>(26-104)", 
               "컴퓨팅 탐색: 실생활에서 활용하기<br>(26-104)", 
               "컴퓨팅 탐색: 실생활에서 활용하기<br>(26-104)", "", "", "", ""]
    }

    df = pd.DataFrame(data)

    colors = {
        "인체생물학": "#f28b82",
        "생명의료윤리": "#81c995",
        "처음 배우는 서양사": "#fbbc04",
        "대중예술의 이해": "#ea4335",
        "제지화학 및 실험": "#46bdc6",
        "컴퓨팅 탐색": "#fdd663"
    }

    def colorize_cell(cell):
        for key, color in colors.items():
            if key in cell:
                return f'<td style="background-color:{color}; text-align:center; vertical-align:middle; color:black; font-weight:bold;">{cell}</td>'
        return f'<td style="text-align:center;">{cell}</td>'

    table_html = "<table style='border-collapse: collapse; width:100%; border:1px solid #ccc;'>"
    table_html += "<tr>" + "".join([f"<th style='border:1px solid #ccc; background:#e8eaed;'>{col}</th>" for col in df.columns]) + "</tr>"

    for _, row in df.iterrows():
        table_html += "<tr>" + "".join([colorize_cell(str(cell)) for cell in row]) + "</tr>"

    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)

    st.subheader("이번 학기 요약 (st.metric)")
    col1, col2 = st.columns(2)
    col1.metric(label="수강 과목 수", value="6")
    col2.metric(label="총 학점", value="18", delta="+3")

    st.write("---")
    st.caption("Streamlit으로 만든 시간표 시각화 예시")

def main():
    st.set_page_config(page_title="나의 포트폴리오", page_icon="💡")

    if "page" not in st.session_state:
        st.session_state.page = "main"

    # 페이지 전환용 콜백
    def go_intro():
        st.session_state.page = "intro"
    def go_timetable():
        st.session_state.page = "timetable"
    def go_main():
        st.session_state.page = "main"

    if st.session_state.page == "main":
        st.title("💡 나의 Streamlit 포트폴리오")
        col1, col2 = st.columns(2)
        col1.button("👨‍🔬 자기소개 페이지 보기", on_click=go_intro)
        col2.button("📚 시간표 페이지 보기", on_click=go_timetable)

    elif st.session_state.page == "intro":
        show_intro()
        st.button("⬅️ 뒤로가기", on_click=go_main)

    elif st.session_state.page == "timetable":
        show_timetable()
        st.button("⬅️ 뒤로가기", on_click=go_main)

main()
