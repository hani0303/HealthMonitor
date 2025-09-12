import screen_brightness_control as sbc
from openai import OpenAI
import os

class EnvironmentOptimizer:
    """사용자 자세와 외부 환경에 따라 최적의 환경을 제안하는 클래스."""
    
    def calculate_viewing_distance(self, face_landmarks, frame_width):
        """얼굴 랜드마크를 이용해 모니터와의 거리를 추정합니다."""
        if not face_landmarks: return 0
        
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        
        eye_dist_pixels = abs(left_eye.x - right_eye.x) * frame_width
        if eye_dist_pixels > 0:
            # 초점 거리 공식을 이용한 간이 거리 계산
            distance_cm = (6.3 * frame_width) / (2 * eye_dist_pixels)
            return min(max(distance_cm, 20), 200)
        return 0

    def calculate_viewing_angle(self, pose_landmarks):
        """목과 머리의 상대적 위치로 시청 각도(거북목)를 추정합니다."""
        if not pose_landmarks: return 0
        
        nose = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
        left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        head_tilt = (nose.y - shoulder_y) * 100 # 수직 차이
        
        return head_tilt * 0.5 # 각도로 변환

    def adjust_brightness_for_eye_health(self, current_hour):
        """시간대에 따라 화면 밝기를 자동 조절합니다."""
        if 6 <= current_hour < 9: target_brightness = 70
        elif 9 <= current_hour < 18: target_brightness = 80
        elif 18 <= current_hour < 21: target_brightness = 60
        else: target_brightness = 40
        
        try:
            sbc.set_brightness(target_brightness)
            return target_brightness
        except Exception:
            return "N/A"

def get_health_advice(context, api_key, **kwargs):
    """측정된 건강 지표에 따라 OpenAI API를 통해 맞춤형 조언을 생성합니다."""
    if not api_key:
        return "API 키가 설정되지 않았습니다. .env 파일을 확인하세요."

    prompts = {
        "heart_rate": f"현재 심박수는 {kwargs.get('hr')} bpm입니다. 이 수치가 의미하는 건강 상태와 간단한 개선 조언을 30자 내외로 해주세요.",
        "stress": f"현재 스트레스 지수는 {kwargs.get('stress')}% 입니다. 스트레스 완화를 위한 즉각적인 행동 팁을 30자 내외로 알려주세요.",
        "spo2": f"현재 산소포화도는 {kwargs.get('spo2')}% 입니다. 이것이 건강에 미치는 영향과 개선 방법을 30자 내외로 알려주세요.",
    }
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 건강 관리 AI 어시스턴트입니다. 사용자의 건강 데이터를 기반으로 간결하고 실용적인 조언을 한국어로 제공하세요."},
                {"role": "user", "content": prompts.get(context, "전반적인 건강 관리 조언을 30자 내외로 해주세요.")}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception:
        return "현재 상태를 잘 유지하며 건강 관리에 신경 써주세요."