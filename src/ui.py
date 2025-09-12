import streamlit as st
import pandas as pd

def setup_page_config():
    """페이지 기본 설정을 구성합니다."""
    st.set_page_config(page_title="AI 통합 건강 모니터링", page_icon="🏥", layout="wide")
    st.title("🏥 AI 기반 통합 건강 모니터링 & 화면 최적화 시스템")

def create_main_layout():
    """애플리케이션의 메인 레이아웃을 생성하고, UI 요소들을 반환합니다."""
    # 상단 제어판
    col_control1, col_control2 = st.columns([1, 3])
    with col_control1:
        run_button = st.button('📹 시스템 시작/중지', type="primary")
    with col_control2:
        status_placeholder = st.empty()

    # 메인 레이아웃
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("📷 실시간 모니터링")
        frame_window = st.empty()
    
    # UI প্লে스홀더 딕셔너리
    ui_elements = {}
    with col2:
        st.subheader("💓 바이탈 사인")
        ui_elements['hr'] = st.empty()
        ui_elements['bp'] = st.empty()
        ui_elements['spo2'] = st.empty()
        ui_elements['rr'] = st.empty()
    with col3:
        st.subheader("🧠 정신 건강 & 자세")
        ui_elements['stress'] = st.empty()
        ui_elements['hrv'] = st.empty()
        ui_elements['distance'] = st.empty()
        ui_elements['angle'] = st.empty()

    # 하단 조언 및 기록 패널
    col4, col5 = st.columns(2)
    with col4:
        st.subheader("🤖 AI 건강 조언")
        ui_elements['advice'] = st.empty()
    with col5:
        st.subheader("📊 측정 기록")
        ui_elements['history'] = st.empty()

    return run_button, status_placeholder, frame_window, ui_elements

def update_metrics(ui, vitals, env):
    """계산된 건강 지표를 화면에 업데이트합니다."""
    hr_color = "🟢" if 60 <= vitals['hr'] <= 100 else "🟡" if 50 < vitals['hr'] < 110 else "🔴"
    ui['hr'].metric("💓 심박수", f"{vitals['hr']} bpm", delta=hr_color)
    
    bp_status = "정상" if vitals['bp_sys'] <= 130 and vitals['bp_dia'] <= 85 else "주의"
    ui['bp'].metric("🩸 혈압", f"{vitals['bp_sys']}/{vitals['bp_dia']}", delta=f"mmHg ({bp_status})")
    
    # ... 다른 지표들도 유사하게 업데이트 ...
    ui['spo2'].metric("🫁 산소포화도", f"{vitals['spo2']}%", delta="🟢" if vitals['spo2'] >= 95 else "🔴")
    ui['rr'].metric("🌬️ 호흡수", f"{vitals['rr']} rpm", delta="🟢" if 12 <= vitals['rr'] <= 20 else "🟡")
    ui['stress'].metric("😰 스트레스", f"{vitals['stress']}%", delta="🟢" if vitals['stress'] <= 30 else "🔴")
    ui['hrv'].metric("💝 심박변이도", f"{vitals['hrv']}", delta="🟢" if vitals['hrv'] >= 50 else "🟡")
    
    if env['dist'] > 0:
        ui['distance'].metric("👁️ 거리", f"{env['dist']:.0f}cm", delta="🟢" if 45 <= env['dist'] <= 75 else "🟡")
    if env['angle'] != 0:
        ui['angle'].metric("📐 각도", f"{env['angle']:.0f}°", delta="🟢" if -5 <= env['angle'] <= 15 else "🟡")

def display_instructions(placeholder):
    """시스템 대기 상태일 때 사용법 안내를 표시합니다."""
    placeholder.info("""
        📋 **시스템 사용법**:
        1. **'시스템 시작' 버튼을 클릭**하세요.
        2. **얼굴이 화면 중앙에 오도록** 위치를 조정하세요.
        3. **15-30초간 안정된 자세**를 유지하세요.
        ...
        ⚠️ **주의사항**: 본 시스템은 건강 참고용이며 정확한 진단은 의료진과 상담하세요.
    """)
