import math
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import check_and_download_models
from config import SWIPE_TIME_WINDOW, SWIPE_COOLDOWN, SWIPE_THRESHOLD

class HandTracker:
    def __init__(self):
        check_and_download_models()
        
        cwd = os.path.dirname(os.path.abspath(__file__))
        hand_model_path = os.path.join(cwd, 'hand_landmarker.task')

        base_options_hand = python.BaseOptions(model_asset_path=hand_model_path)
        options_hand = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            num_hands=1)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)
        
        # Variables Swipe
        self.hand_history = [] 
        self.last_swipe_time = 0
        self.last_swipe_direction = None
        self.last_swipe_display_time = 0

    def detect(self, mp_image):
        try:
            return self.hand_landmarker.detect(mp_image)
        except Exception as e:
            print(f"[HandLandmarker] error: {e}")
            return None

    def detect_hand_gesture(self, landmarks):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        pinky_mcp = landmarks[17]
        
        def is_extended(tip, joint):
            return math.hypot(tip.x - wrist.x, tip.y - wrist.y) > math.hypot(joint.x - wrist.x, joint.y - wrist.y)

        thumb_ext = is_extended(thumb_tip, landmarks[2])
        index_ext = is_extended(index_tip, landmarks[6])
        middle_ext = is_extended(middle_tip, landmarks[10])
        ring_ext = is_extended(ring_tip, landmarks[14])
        pinky_ext = is_extended(pinky_tip, landmarks[18])

        dist_thumb_pinky = math.hypot(thumb_tip.x - pinky_mcp.x, thumb_tip.y - pinky_mcp.y)
        if dist_thumb_pinky < 0.15: thumb_ext = False
        pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        
        dx = middle_tip.x - wrist.x 
        dy = middle_tip.y - wrist.y 
        roll_angle = math.degrees(math.atan2(dx, -dy))

        if pinch_dist < 0.05: return "PINCH", index_tip, roll_angle
        if thumb_ext and index_ext and middle_ext and not ring_ext and not pinky_ext: return "THREE_FINGERS", index_tip, roll_angle
        if index_ext and not middle_ext and not ring_ext and not pinky_ext: return "POINT", index_tip, roll_angle
        if index_ext and middle_ext and not ring_ext and not pinky_ext: return "TWO_FINGERS", index_tip, roll_angle
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext: return "FIST", None, roll_angle
        if index_ext and middle_ext and ring_ext and pinky_ext: return "OPEN_PALM", None, roll_angle
        return "NONE", None, roll_angle

    def detect_swipe(self, position):
        """Détection de mouvement brusque (Swipe)"""
        current_time = time.time()
        # Ajouter la position actuelle (x)
        self.hand_history.append((current_time, position.x))
        
        # Nettoyer l'historique (garder seulement les dernières X secondes)
        self.hand_history = [h for h in self.hand_history if current_time - h[0] < SWIPE_TIME_WINDOW]
        
        if len(self.hand_history) < 2:
            return None
            
        if current_time - self.last_swipe_time < SWIPE_COOLDOWN:
            return None

        # Calculer le déplacement
        start_x = self.hand_history[0][1]
        end_x = self.hand_history[-1][1]
        dx = end_x - start_x
        
        if abs(dx) > SWIPE_THRESHOLD:
            self.last_swipe_time = current_time
            self.hand_history = [] # Reset
            direction = "RIGHT" if dx > 0 else "LEFT"
            self.last_swipe_direction = direction
            self.last_swipe_display_time = current_time
            return direction
            
        return None

    def close(self):
        if self.hand_landmarker:
            self.hand_landmarker.close()
