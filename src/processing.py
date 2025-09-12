import numpy as np
from scipy import signal
from scipy.signal import find_peaks

class VitalSignsProcessor:
    """PPG 신호로부터 다양한 건강 지표를 계산하는 클래스."""
    
    def calculate_heart_rate(self, green_buffer, fs=30):
        """PPG 신호(Green 채널)에서 심박수(BPM)를 계산합니다."""
        if len(green_buffer) < 150: return 0
        
        signal_data = np.array(green_buffer) - np.mean(green_buffer)
        
        nyquist = fs / 2
        low, high = 0.7 / nyquist, 4.0 / nyquist
        
        if len(signal_data) > 60:
            sos = signal.butter(4, [low, high], btype='band', output='sos')
            filtered_signal = signal.sosfilt(sos, signal_data)
            
            fft_data = np.fft.fft(filtered_signal)
            freqs = np.fft.fftfreq(len(filtered_signal), 1/fs)
            
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft_data[:len(fft_data)//2])
            
            valid_range = (positive_freqs >= 1) & (positive_freqs <= 3)
            if np.any(valid_range):
                peak_freq = positive_freqs[valid_range][np.argmax(positive_fft[valid_range])]
                heart_rate = int(peak_freq * 60)
                return max(50, min(heart_rate, 180))
        return 0

    def estimate_blood_pressure(self, green_buffer, heart_rate):
        """PPG 패턴과 심박수를 이용해 혈압을 추정합니다. (의료용 아님)"""
        if len(green_buffer) < 150: return 0, 0
        
        signal_data = np.array(green_buffer) - np.mean(green_buffer)
        peaks, _ = find_peaks(signal_data, height=np.std(signal_data)*0.3, distance=15)
        
        if len(peaks) > 3:
            hrv = np.std(np.diff(peaks))
            hr_adj = (heart_rate - 70) * 0.5
            hrv_adj = max(0, (10 - hrv)) * 2
            
            systolic = int(120 + hr_adj + hrv_adj)
            diastolic = int(80 + hr_adj * 0.3 + hrv_adj * 0.5)
            
            return max(90, min(systolic, 200)), max(60, min(diastolic, 120))
        return 0, 0

    def calculate_spo2(self, red_buffer, blue_buffer):
        """RGB 신호 비율을 분석하여 산소포화도를 추정합니다."""
        if len(red_buffer) < 100 or len(blue_buffer) < 100: return 0
        
        red_ac, red_dc = np.std(red_buffer), np.mean(red_buffer)
        blue_ac, blue_dc = np.std(blue_buffer), np.mean(blue_buffer)
        
        if red_dc > 0 and blue_dc > 0:
            ratio = (red_ac / red_dc) / (blue_ac / blue_dc)
            spo2 = int(110 - 25 * ratio)
            return max(85, min(spo2, 100))
        return 0

    def calculate_respiratory_rate(self, green_buffer, fs=30):
        """PPG의 저주파 성분을 분석하여 호흡수를 계산합니다."""
        if len(green_buffer) < 200: return 0
        
        signal_data = np.array(green_buffer) - np.mean(green_buffer)
        
        nyquist = fs / 2
        low, high = 0.1 / nyquist, 0.5 / nyquist
        
        if len(signal_data) > 100:
            sos = signal.butter(4, [low, high], btype='band', output='sos')
            res_signal = signal.sosfilt(sos, signal_data)
            
            peaks, _ = find_peaks(res_signal, height=np.std(res_signal)*0.2, distance=30)
            if len(peaks) > 2:
                duration_min = len(signal_data) / (fs * 60)
                res_rate = int(len(peaks) / duration_min)
                return max(8, min(res_rate, 30))
        return 0

    def calculate_stress_hrv(self, green_buffer, fs=30):
        """심박변이도(HRV)를 분석하여 스트레스 지수를 계산합니다."""
        if len(green_buffer) < 200: return 0, 0
        
        signal_data = np.array(green_buffer) - np.mean(green_buffer)
        peaks, _ = find_peaks(signal_data, height=np.std(signal_data)*0.3, distance=15)
        
        if len(peaks) > 10:
            rr_intervals = np.diff(peaks) * (1000 / fs)
            if len(rr_intervals) > 5:
                rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
                sdnn = np.std(rr_intervals)
                
                hrv_score = int(min(100, (rmssd + sdnn) / 2))
                stress_level = int(max(0, 100 - hrv_score))
                return stress_level, hrv_score
        return 0, 0
