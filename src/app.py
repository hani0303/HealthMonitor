import streamlit as st
import cv2
import time
import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# 분리된 모듈 임포트
from ui import setup_page_config, create_main_layout, update_metrics, display_instructions
from vision import VisionManager
from processing import VitalSignsProcessor
from utils import EnvironmentOptimizer, get_health_advice

def main():
    # --- 1. 초기 설정 ---
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    setup_page_config()

    # 객체 초기화
    vision = VisionManager()
    processor = VitalSignsProcessor()
    optimizer = EnvironmentOptimizer()

    # 세션 상태 초기화
    if 'run' not in st.session_state: st.session_state.run = False
    if 'measurement_data' not in st.session_state: st.session_state.measurement_data = []
    
    # --- 2. UI 레이아웃 생성 ---
    run_button, status_placeholder, frame_window, ui_elements = create_main_layout()

    if run_button:
        st.session_state.run = not st.session_state.run

    # --- 3. 메인 루프 ---
    if st.session_state.run:
        status_placeholder.success("🟢 시스템 실행 중...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        last_measurement_time = time.time()

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            # 비전 처리
            pose_results, face_results = vision.process_frame(frame)
            vision.extract_ppg_signal(frame, face_results)
            vision.draw_landmarks(frame, pose_results)
            
            # 5초마다 건강 지표 계산 및 UI 업데이트
            if time.time() - last_measurement_time >= 5.0:
                # 신호 처리
                vitals = {
                    'hr': processor.calculate_heart_rate(vision.green_buffer),
                    'spo2': processor.calculate_spo2(vision.red_buffer, vision.blue_buffer),
                    'rr': processor.calculate_respiratory_rate(vision.green_buffer)
                }
                vitals['bp_sys'], vitals['bp_dia'] = processor.estimate_blood_pressure(vision.green_buffer, vitals['hr'])
                vitals['stress'], vitals['hrv'] = processor.calculate_stress_hrv(vision.green_buffer)

                # 환경 분석
                env = {
                    'dist': optimizer.calculate_viewing_distance(face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None, frame.shape[1]),
                    'angle': optimizer.calculate_viewing_angle(pose_results.pose_landmarks if pose_results else None)
                }
                
                # UI 업데이트
                update_metrics(ui_elements, vitals, env)
                
                # AI 조언 생성
                # ... (가장 주의가 필요한 항목에 대한 조언 생성 로직)
                advice_text = get_health_advice("stress", api_key, stress=vitals['stress']) if vitals['stress'] > 60 else "모든 지표가 안정적입니다. 😊"
                ui_elements['advice'].info(f"🤖 **AI 조언**: {advice_text}")

                # 기록 저장 및 표시
                # ...
                
                last_measurement_time = time.time()
            
            frame_window.image(frame, channels="BGR")
            time.sleep(0.02) # 약 30fps 유지

        cap.release()
        status_placeholder.info("⚫ 시스템 대기 중...")
    else:
        display_instructions(frame_window)

if __name__ == "__main__":
    main()
