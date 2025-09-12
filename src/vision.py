import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class VisionManager:
    """카메라 프레임으로부터 비전 기반 데이터를 추출하는 클래스."""
    
    def __init__(self, buffer_size=300):
        # Mediapipe 모델 초기화
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5, model_complexity=0)
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        
        # PPG 신호 저장을 위한 버퍼
        self.red_buffer = deque(maxlen=buffer_size)
        self.green_buffer = deque(maxlen=buffer_size)
        self.blue_buffer = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)

    def process_frame(self, frame):
        """단일 프레임을 처리하여 얼굴 및 자세 랜드마크를 반환합니다."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(image_rgb)
        face_results = self.face_mesh.process(image_rgb)
        return pose_results, face_results

    def extract_ppg_signal(self, frame, face_landmarks):
        """얼굴 ROI(이마)에서 PPG 신호를 추출하여 버퍼에 저장합니다."""
        if not face_landmarks or not face_landmarks.multi_face_landmarks:
            return
        
        h, w = frame.shape[:2]
        landmarks = face_landmarks.multi_face_landmarks[0].landmark
        
        forehead_indices = [10, 151, 9, 8] # 이마 주변 랜드마크 인덱스
        forehead_coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in forehead_indices]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        forehead_hull = cv2.convexHull(np.array(forehead_coords))
        cv2.fillPoly(mask, [forehead_hull], 255)
        
        roi_pixels = frame[mask > 0]
        if len(roi_pixels) > 0:
            self.red_buffer.append(np.mean(roi_pixels[:, 2]))
            self.green_buffer.append(np.mean(roi_pixels[:, 1]))
            self.blue_buffer.append(np.mean(roi_pixels[:, 0]))
            self.timestamps.append(time.time())
            # 시각화를 위해 ROI 그리기
            cv2.polylines(frame, [forehead_hull], True, (0, 255, 0), 2)
    
    def draw_landmarks(self, frame, pose_results):
        """프레임에 포즈 랜드마크를 그립니다."""
        if pose_results and pose_results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
