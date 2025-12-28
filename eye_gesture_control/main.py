import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
import os
import json
import threading
import queue
import urllib.request
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from screeninfo import get_monitors
import pygetwindow as gw

# --- CONFIGURATION GLOBALE ---
CONFIG_FILE = "screen_calibration.json"
# Note sur Alpha : 1.0 = aucune stabilisation (brut), 0.1 = très fort lissage (lent)
MOUSE_SMOOTHING = 0.4  
HEAD_SMOOTHING = 0.8
SCROLL_SPEED = 30
SWIPE_THRESHOLD = 0.04 # Sensibilité augmentée (0.05 -> 0.04)
SWIPE_TIME_WINDOW = 0.6
SWIPE_COOLDOWN = 1.0
VOLUME_DEBOUNCE = 0.2
AFK_TIMEOUT = 5.0
PAUSE_COOLDOWN = 1.5  # Temps min entre deux actions

# --- PALETTE "CYBER-GLASS" (BGR) ---
C_CYAN_DIM = (100, 100, 0)
C_CYAN_BRIGHT = (255, 255, 100)
C_MAGENTA_DEEP = (100, 0, 100)
C_MAGENTA_NEON = (255, 0, 255)
C_BLUE_TECH = (255, 150, 0)
C_GREEN_DATA = (50, 255, 100)
C_ALERT_RED = (0, 0, 255)
C_WHITE = (240, 240, 255)
C_GLASS_DARK = (10, 15, 20)

# --- CLASSES UTILITAIRES ---

class Stabilizer:
    """Lisse les valeurs numériques pour éviter le jittering (tremblements)"""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

class ActionWorker(threading.Thread):
    """Exécute les actions PyAutoGUI dans un thread séparé"""
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self.running = True
        self.daemon = True

# --- CLASSE PRINCIPALE ---
    def run(self):
        while self.running:
            try:
                item = self.queue.get(timeout=0.5)
                if item is None:
                    # Sentinel de stop
                    self.queue.task_done()
                    break

                action_type, args = item

                if action_type == "move":
                    pyautogui.moveTo(*args)
                elif action_type == "click":
                    pyautogui.click()
                elif action_type == "scroll":
                    pyautogui.scroll(*args)
                elif action_type == "key":
                    pyautogui.press(*args)
                elif action_type == "hotkey":
                    pyautogui.hotkey(*args)
                elif action_type == "custom":
                    func, f_args = args
                    func(*f_args)

                self.queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[WORKER ERROR] {e}")

    def add_action(self, action_type, args=None):
        with self.queue.mutex:
            if action_type == "move":
                self.queue.queue = deque([item for item in self.queue.queue if item and item[0] != "move"])
            elif action_type == "scroll":
                # Option robuste anti-inertie : purge tous les scroll en attente
                self.queue.queue = deque([item for item in self.queue.queue if item and item[0] != "scroll"])

        self.queue.put((action_type, args))

    def stop(self):
        # Stop immédiat
        self.running = False
        try:
            self.queue.put(None)  # Sentinel
        except Exception:
            pass
        self.join(timeout=2.0)

class EyeGestureController:
    def __init__(self):
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        self.check_and_download_models()
        
        # --- MediaPipe Setup ---
        cwd = os.path.dirname(os.path.abspath(__file__))
        face_model_path = os.path.join(cwd, 'face_landmarker.task')
        hand_model_path = os.path.join(cwd, 'hand_landmarker.task')

        base_options_face = python.BaseOptions(model_asset_path=face_model_path)
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options_face)

        base_options_hand = python.BaseOptions(model_asset_path=hand_model_path)
        options_hand = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            num_hands=1)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)
        
        # --- Worker ---
        self.worker = ActionWorker()
        self.worker.start()

        # --- Caméra ---
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # --- Optimisation Vignettage ---
        self.vignette_map = None
        # On initialise une première fois avec la résolution cible par défaut
        self._generate_vignette_mask(640, 480)

        # --- Ecrans & Calibration ---
        self.monitors = get_monitors()
        self.screen_centers = {
            "ECRAN_1_GAUCHE": {"yaw": -20, "pitch": 5, "monitor_idx": 0},
            "ECRAN_2_CENTRE": {"yaw": 0, "pitch": 5, "monitor_idx": 0},
            "ECRAN_3_HAUT":   {"yaw": 0, "pitch": 20, "monitor_idx": 0},
            "ECRAN_4_DROITE": {"yaw": 20, "pitch": 5, "monitor_idx": 0}
        }
        
        if len(self.monitors) > 1:
            for i, key in enumerate(self.screen_centers.keys()):
                if i < len(self.monitors):
                    self.screen_centers[key]["monitor_idx"] = i

        self.load_calibration()
        self.calibration_mode = False
        self.calibration_step = 0
        self.calibration_keys = list(self.screen_centers.keys())

        # --- État Système ---
        self.current_screen = "INCONNU"
        self.active_monitor = self.monitors[0] if self.monitors else None
        
        # Stabilisateurs
        self.yaw_stabilizer = Stabilizer(HEAD_SMOOTHING)
        self.pitch_stabilizer = Stabilizer(HEAD_SMOOTHING)
        self.mouse_stabilizer_x = Stabilizer(MOUSE_SMOOTHING)
        self.mouse_stabilizer_y = Stabilizer(MOUSE_SMOOTHING)
        
        self.last_action_time = {
            "volume": 0, "swipe": 0, "typing": 0, "pause": 0
        }
        self.last_toggle_time = 0
        self.fist_state_locked = False # Verrouillage pour éviter le rebond play/pause
        self.fist_release_start_time = 0 # Timer pour valider le relâchement du poing
        self.last_log_message = "SYSTEM INITIALIZED"
        self.last_log_time = time.time()
        
        # Variables Swipe
        self.hand_history = [] 
        self.last_swipe_time = 0
        self.last_swipe_direction = None
        self.last_swipe_display_time = 0

        self.last_face_detected_time = time.time()
        self.is_afk = False
        self.mouse_paused = False

        print(f"Système initialisé. {len(self.monitors)} moniteurs détectés.")

    def _generate_vignette_mask(self, width, height):
        """Génère le masque de vignettage une seule fois pour économiser le CPU"""
        try:
            kernel_x = cv2.getGaussianKernel(width, 200)
            kernel_y = cv2.getGaussianKernel(height, 200)
            kernel = kernel_y * kernel_x.T
            mask = 255 * kernel / np.linalg.norm(kernel)
            mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            # 0.6 luminosité min (bords), 1.0 max (centre)
            self.vignette_map = (0.6 + 0.4 * mask[:, :, np.newaxis]).astype(np.float32)
            print(f"Vignette générée pour {width}x{height}")
        except Exception as e:
            print(f"Erreur génération vignette: {e}")
            self.vignette_map = None

    def check_and_download_models(self):
        models = {
            "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        }
        for filename, url in models.items():
            if not os.path.exists(filename):
                try:
                    urllib.request.urlretrieve(url, filename)
                except Exception as e:
                    print(f"Erreur téléchargement {filename}: {e}")
                    raise

    def log_action(self, message):
        print(f"[ACTION] {message}")
        self.last_log_message = message
        self.last_log_time = time.time()

    def load_calibration(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    for key, val in data.items():
                        if key in self.screen_centers:
                            self.screen_centers[key].update(val)
            except Exception as e:
                print(f"Erreur chargement calibration: {e}")

    def save_calibration(self):
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.screen_centers, f, indent=4)
        except Exception as e:
            print(f"Erreur sauvegarde calibration: {e}")

    def calibrate_step(self, yaw, pitch):
        if self.calibration_step < len(self.calibration_keys):
            screen_name = self.calibration_keys[self.calibration_step]
            self.screen_centers[screen_name]["yaw"] = yaw
            self.screen_centers[screen_name]["pitch"] = pitch
            self.log_action(f"CALIB SAVED: {screen_name}")
            self.calibration_step += 1
            time.sleep(0.5)
        
        if self.calibration_step >= len(self.calibration_keys):
            self.calibration_mode = False
            self.save_calibration()
            self.log_action("CALIBRATION COMPLETE")

    def get_head_pose(self, image_shape, landmarks):
        img_h, img_w, _ = image_shape
        face_2d = []
        face_3d = []
        key_points = [1, 199, 33, 263, 61, 291]

        for idx in key_points:
            lm = landmarks[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        focal_length = 1 * img_w
        
        # CORRECTIF MATRICE CAMERA (Principal Point = Width/2, Height/2)
        cam_matrix = np.array([[focal_length, 0, img_w / 2],
                               [0, focal_length, img_h / 2],
                               [0, 0, 1]])
                               
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles[1] * 360, angles[0] * 360

    def determine_screen(self, yaw, pitch):
        best_screen = self.current_screen
        min_dist = float('inf')
        hysteresis_bonus = 5.0 

        for screen_name, data in self.screen_centers.items():
            c_yaw = data["yaw"]
            c_pitch = data["pitch"]
            dist = math.sqrt((yaw - c_yaw)**2 + (pitch - c_pitch)**2)
            if screen_name == self.current_screen:
                dist -= hysteresis_bonus
            if dist < min_dist:
                min_dist = dist
                best_screen = screen_name
        return best_screen

    def update_active_monitor(self, screen_name):
        if screen_name in self.screen_centers:
            idx = self.screen_centers[screen_name].get("monitor_idx", 0)
            if idx < len(self.monitors):
                self.active_monitor = self.monitors[idx]

    # --- ACTIONS SYSTEME ---
    def trigger_play_pause_logic(self):
        """Active la fenêtre sous le curseur et envoie Espace (Version Robuste)"""
        if not self.active_monitor: return
        center_x = self.active_monitor.x + self.active_monitor.width // 2
        center_y = self.active_monitor.y + self.active_monitor.height // 2
        
        try:
            # 1. Essai de trouver la fenêtre intelligemment (sans cliquer aveuglément)
            windows = gw.getWindowsAt(center_x, center_y)
            target_window = None
            if windows:
                for w in windows:
                    # On ignore notre propre fenêtre de debug/caméra
                    if "Eye" not in w.title and "Gesture" not in w.title and w.title != "":
                        target_window = w
                        break
            
            if target_window:
                try:
                    if target_window.isMinimized: target_window.restore()
                    target_window.activate()
                except: pass
                # Petite pause pour laisser le temps au focus de se faire
                time.sleep(0.1) 
                pyautogui.press('space')
            else:
                # 2. Fallback : Clic pour focus puis Espace
                pyautogui.click(center_x, center_y)
                # Important : délai pour ne pas que le clic+espace fassent un double-toggle
                time.sleep(0.15) 
                pyautogui.press('space')
                
        except Exception as e:
            print(f"Fallback Pause: {e}")
            pyautogui.press('space')

    def trigger_play_pause(self):
        # Le cooldown est géré par la logique "fist_state_locked" mais on garde une sécu
        if time.time() - self.last_toggle_time < PAUSE_COOLDOWN:
            return
        
        self.log_action("MEDIA TOGGLE (PAUSE/PLAY)")
        self.worker.add_action("custom", (self.trigger_play_pause_logic, []))
        self.last_toggle_time = time.time()

    # --- DETECTION GESTES ---
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

    def process_actions(self, gesture, position, roll_angle, image=None):
        current_time = time.time()
        
        # Actions Ecrans
        if self.current_screen == "ECRAN_3_HAUT":
            # LOGIQUE ROBUSTE DE VERROUILLAGE/DEVERROUILLAGE (Anti-Rebond)
            if gesture == "FIST":
                self.fist_release_start_time = 0 # Annule tout timer de déverrouillage
                
                # On déclenche seulement si non verrouillé ET cooldown passé
                if not self.fist_state_locked and (current_time - self.last_toggle_time > PAUSE_COOLDOWN):
                    self.trigger_play_pause()
                    self.fist_state_locked = True
            
            else:
                # Si le geste n'est PAS FIST (donc NONE, PALM, etc.)
                if self.fist_state_locked:
                    # On lance le chrono de relâchement si pas déjà fait
                    if self.fist_release_start_time == 0:
                        self.fist_release_start_time = current_time
                    
                    # SI et SEULEMENT SI le geste reste "PAS FIST" pendant 0.5s, on déverrouille
                    elif current_time - self.fist_release_start_time > 0.5:
                        self.fist_state_locked = False
                        self.fist_release_start_time = 0
            return
        
        # Si on change d'écran, on reset le lock pour ne pas rester coincé
        self.fist_state_locked = False
        self.fist_release_start_time = 0

        if self.current_screen == "ECRAN_1_GAUCHE":
            is_horizontal = abs(roll_angle) > 30
            if gesture == "TWO_FINGERS" and position and is_horizontal:
                swipe = self.detect_swipe(position)
                if swipe:
                    if swipe == "RIGHT":
                        self.worker.add_action("hotkey", ('ctrl', 'pagedown'))
                        self.log_action("Tab Next >>")
                    else:
                        self.worker.add_action("hotkey", ('ctrl', 'pageup'))
                        self.log_action("<< Tab Prev")
                return

        if gesture == "THREE_FINGERS" and position:
            DEADZONE = 15
            speed = 0
            if roll_angle > DEADZONE:
                intensity = (roll_angle - DEADZONE) / 5
                speed = -int(SCROLL_SPEED * intensity)
            elif roll_angle < -DEADZONE:
                intensity = (abs(roll_angle) - DEADZONE) / 5
                speed = int(SCROLL_SPEED * intensity)
            
            if speed != 0:
                self.worker.add_action("scroll", (speed,))
                if image is not None:
                    # FEEDBACK VISUEL RESTAURÉ (Joystick)
                    h, w, _ = image.shape
                    cx, cy = int(position.x * w), int(position.y * h)
                    color_joy = (0, 255, 255) # Jaune/Cyan
                    if abs(roll_angle) > DEADZONE: color_joy = (0, 255, 0) # Vert quand actif
                    
                    # Cercle Extérieur
                    cv2.circle(image, (cx, cy), 45, color_joy, 2)
                    
                    # Ligne Directionnelle
                    rad = math.radians(roll_angle - 90)
                    ex = int(cx + 45 * math.cos(rad))
                    ey = int(cy + 45 * math.sin(rad))
                    cv2.line(image, (cx, cy), (ex, ey), color_joy, 2)
                    
                    # Texte
                    cv2.putText(image, f"SCROLL {speed}", (cx-35, cy-55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_joy, 1)
            return

        if gesture == "POINT" and position and self.active_monitor and not self.mouse_paused:
            # Clamping pour éviter l'extrapolation hors bornes
            px = min(max(position.x, 0.1), 0.9)
            py = min(max(position.y, 0.1), 0.9)
            
            screen_x_rel = np.interp(px, [0.1, 0.9], [0, self.active_monitor.width])
            screen_y_rel = np.interp(py, [0.1, 0.9], [0, self.active_monitor.height])
            
            target_x = self.active_monitor.x + screen_x_rel
            target_y = self.active_monitor.y + screen_y_rel
            
            target_x = max(self.active_monitor.x, min(target_x, self.active_monitor.x + self.active_monitor.width - 1))
            target_y = max(self.active_monitor.y, min(target_y, self.active_monitor.y + self.active_monitor.height - 1))

            # Application du lissage souris (Stabilizer)
            smooth_x = self.mouse_stabilizer_x.update(target_x)
            smooth_y = self.mouse_stabilizer_y.update(target_y)

            # Envoi INT pour éviter les problèmes de drivers
            self.worker.add_action("move", (int(smooth_x), int(smooth_y)))

    # --- NOUVEAU MOTEUR DE RENDU VISUEL "GLASS & NEON" ---

    def draw_glass_panel(self, img, pt1, pt2, color=C_GLASS_DARK, alpha=0.6):
        """Dessine un panneau style 'verre' semi-transparent"""
        overlay = img.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        # Bordure fine
        cv2.rectangle(overlay, pt1, pt2, (color[0]+30, color[1]+30, color[2]+30), 1)
        # Coins Tech
        x1, y1 = pt1
        x2, y2 = pt2
        l = 10
        c_corn = C_CYAN_DIM
        cv2.line(img, (x1, y1), (x1+l, y1), c_corn, 1)
        cv2.line(img, (x1, y1), (x1, y1+l), c_corn, 1)
        cv2.line(img, (x2, y2), (x2-l, y2), c_corn, 1)
        cv2.line(img, (x2, y2), (x2, y2-l), c_corn, 1)
        
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def draw_neon_text(self, img, text, pos, scale, color, thickness=1):
        x, y = pos
        # Glow artificiel (plus performant que blur)
        cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+1) # Ombre
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def draw_tech_circle(self, img, center, radius, color, thickness=1, segments=4):
        """Dessine un cercle segmenté qui tourne"""
        angle_offset = (time.time() * 50) % 360
        seg_len = 360 // segments
        for i in range(segments):
            start_angle = i * seg_len + angle_offset
            end_angle = start_angle + (seg_len * 0.6)
            cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, thickness)

    def draw_hud_header(self, image, fps):
        h, w, _ = image.shape
        # Top Bar Background
        self.draw_glass_panel(image, (0, 0), (w, 40), C_GLASS_DARK, 0.7)
        
        # Titre & FPS
        self.draw_neon_text(image, "EYE.OS // V2.4", (20, 25), 0.6, C_CYAN_BRIGHT, 2)
        
        # Indicateur Pulse (Battement de coeur du système)
        pulse = abs(math.sin(time.time() * 3))
        col_pulse = (0, int(255*pulse), 0)
        cv2.circle(image, (w-120, 20), 4, col_pulse, -1)
        self.draw_neon_text(image, f"{int(fps)} FPS", (w-100, 25), 0.5, C_GREEN_DATA)

    def draw_hud_sidebar(self, image, yaw, pitch):
        h, w, _ = image.shape
        # Panneau Gauche: Logs
        self.draw_glass_panel(image, (10, h-100), (300, h-10), C_GLASS_DARK, 0.5)
        self.draw_neon_text(image, f">> {self.last_log_message}", (20, h-50), 0.45, C_WHITE)
        # Timer
        uptime = int(time.time() - self.start_time)
        m, s = divmod(uptime, 60)
        self.draw_neon_text(image, f"T+{m:02}:{s:02}", (20, h-25), 0.4, C_CYAN_DIM)

        # Panneau Droite: Radar
        radar_size = 140
        rx, ry = w - 160, h - 160
        self.draw_glass_panel(image, (w-180, h-180), (w-10, h-10), C_GLASS_DARK, 0.5)
        
        # Radar Crosshair
        cx, cy = w-95, h-95
        cv2.line(image, (cx-60, cy), (cx+60, cy), C_CYAN_DIM, 1)
        cv2.line(image, (cx, cy-60), (cx, cy+60), C_CYAN_DIM, 1)
        cv2.circle(image, (cx, cy), 40, C_CYAN_DIM, 1)
        
        # Radar Gaze Dot (Cible)
        gx = int(cx + (yaw * 1.5))
        gy = int(cy - (pitch * 1.5))
        cv2.line(image, (cx, cy), (gx, gy), C_MAGENTA_DEEP, 1)
        cv2.circle(image, (gx, gy), 4, C_MAGENTA_NEON, -1)
        cv2.circle(image, (gx, gy), 8, C_MAGENTA_NEON, 1)

    def draw_face_hud(self, image, landmarks):
        """Dessine un cadre de verrouillage autour du visage"""
        h, w, _ = image.shape
        
        # Calcul Bounding Box approx
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x1, y1 = int(min(xs)*w), int(min(ys)*h)
        x2, y2 = int(max(xs)*w), int(max(ys)*h)
        
        # Marge
        pad = 20
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = min(w, x2+pad), min(h, y2+pad)
        
        # Couleur dynamique selon état
        color = C_ALERT_RED if self.is_afk else C_CYAN_BRIGHT
        if self.calibration_mode: color = C_MAGENTA_NEON

        # Coins du cadre (Bracket style)
        sl = 20 # segment length
        # Haut Gauche
        cv2.line(image, (x1, y1), (x1+sl, y1), color, 2)
        cv2.line(image, (x1, y1), (x1, y1+sl), color, 2)
        # Haut Droite
        cv2.line(image, (x2, y1), (x2-sl, y1), color, 2)
        cv2.line(image, (x2, y1), (x2, y1+sl), color, 2)
        # Bas Gauche
        cv2.line(image, (x1, y2), (x1+sl, y2), color, 2)
        cv2.line(image, (x1, y2), (x1, y2-sl), color, 2)
        # Bas Droite
        cv2.line(image, (x2, y2), (x2-sl, y2), color, 2)
        cv2.line(image, (x2, y2), (x2, y2-sl), color, 2)
        
        # Scan Line Effect (Ligne qui descend)
        scan_h = int(y1 + (time.time() * 200) % (y2-y1))
        cv2.line(image, (x1, scan_h), (x2, scan_h), color, 1)

    def draw_status_overlay(self, image):
        """Affiche les écrans/modes au centre si changement"""
        h, w, _ = image.shape
        cx, cy = w//2, h//2
        
        # Affichage Ecran Actif en Haut
        screen_label = self.current_screen.replace("ECRAN_", "").replace("_", " ")
        self.draw_glass_panel(image, (cx-100, 50), (cx+100, 80), C_GLASS_DARK, 0.8)
        self.draw_neon_text(image, screen_label, (cx-90, 72), 0.6, C_BLUE_TECH, 1)

        # Mode Calibration
        if self.calibration_mode:
            cv2.rectangle(image, (0,0), (w,h), C_MAGENTA_DEEP, 20) # Bordure massive
            self.draw_neon_text(image, "CALIBRATION ENGAGED", (cx-150, cy), 1.0, C_MAGENTA_NEON, 2)
            self.draw_neon_text(image, "LOOK AT TARGET & PRESS SPACE", (cx-180, cy+40), 0.6, C_WHITE, 1)

        # Mode AFK
        if self.is_afk:
            overlay = image.copy()
            cv2.rectangle(overlay, (0,0), (w,h), (0,0,50), -1) # Fond rouge sombre
            cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
            self.draw_neon_text(image, "SYSTEM LOCKED / AFK", (cx-140, cy), 0.8, C_ALERT_RED, 2)
            
        # Feedback Swipe
        if time.time() - self.last_swipe_display_time < 1.0 and self.last_swipe_direction:
            text = f"SWIPE {self.last_swipe_direction} >>"
            self.draw_glass_panel(image, (cx-100, cy+100), (cx+100, cy+140), C_GLASS_DARK, 0.8)
            self.draw_neon_text(image, text, (cx-80, cy+128), 0.7, C_GREEN_DATA)

    def shutdown(self):
        """Libère proprement toutes les ressources (caméra, fenêtres, threads, mediapipe)."""
        # Worker
        try:
            if hasattr(self, "worker") and self.worker:
                self.worker.stop()
        except Exception as e:
            print(f"[SHUTDOWN] worker.stop error: {e}")

        # Caméra
        try:
            if hasattr(self, "cap") and self.cap:
                self.cap.release()
        except Exception as e:
            print(f"[SHUTDOWN] cap.release error: {e}")

        # OpenCV windows
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[SHUTDOWN] destroyAllWindows error: {e}")

        # MediaPipe landmarkers (IMPORTANT)
        try:
            if hasattr(self, "face_landmarker") and self.face_landmarker:
                self.face_landmarker.close()
        except Exception as e:
            print(f"[SHUTDOWN] face_landmarker.close error: {e}")

        try:
            if hasattr(self, "hand_landmarker") and self.hand_landmarker:
                self.hand_landmarker.close()
        except Exception as e:
            print(f"[SHUTDOWN] hand_landmarker.close error: {e}")

    def run(self):
        prev_time = time.time()

        try:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    continue

                self.frame_count += 1

                # FPS Calculation
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30
                prev_time = curr_time

                image = cv2.flip(image, 1)

                # --- VIGNETTAGE OPTIMISÉ ---
                if self.vignette_map is None or image.shape[:2] != self.vignette_map.shape[:2]:
                    h, w = image.shape[:2]
                    self._generate_vignette_mask(w, h)

                if self.vignette_map is not None:
                    image = (image * self.vignette_map).astype(np.uint8)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                )

                # --- Head Tracking (protégé) ---
                try:
                    face_result = self.face_landmarker.detect(mp_image)
                except KeyboardInterrupt:
                    # Sortie immédiate mais propre
                    raise
                except Exception as e:
                    print(f"[FaceLandmarker] error: {e}")
                    face_result = None

                yaw_stable, pitch_stable = 0, 0

                if face_result and face_result.face_landmarks:
                    self.last_face_detected_time = time.time()
                    if self.is_afk:
                        self.log_action("USER IDENTIFIED")
                        self.trigger_play_pause()
                        self.is_afk = False

                    landmarks = face_result.face_landmarks[0]
                    self.draw_face_hud(image, landmarks)

                    raw_yaw, raw_pitch = self.get_head_pose(image.shape, landmarks)
                    yaw_stable = self.yaw_stabilizer.update(raw_yaw)
                    pitch_stable = self.pitch_stabilizer.update(raw_pitch)

                    if not self.calibration_mode:
                        new_screen = self.determine_screen(yaw_stable, pitch_stable)
                        if new_screen != self.current_screen:
                            self.current_screen = new_screen
                            self.update_active_monitor(new_screen)

                            # Reset stabilizers pour éviter le saut de curseur
                            self.mouse_stabilizer_x.value = None
                            self.mouse_stabilizer_y.value = None

                            # Reset lock poing UNIQUEMENT au changement d'écran
                            self.fist_state_locked = False
                            self.fist_release_start_time = 0

                else:
                    if not self.is_afk and (time.time() - self.last_face_detected_time > AFK_TIMEOUT):
                        self.log_action("NO USER - LOCKING")
                        self.trigger_play_pause()
                        self.is_afk = True

                # --- Hand Tracking (protégé) ---
                try:
                    hand_result = self.hand_landmarker.detect(mp_image)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"[HandLandmarker] error: {e}")
                    hand_result = None

                if hand_result and hand_result.hand_landmarks and not self.calibration_mode:
                    landmarks = hand_result.hand_landmarks[0]
                    gesture, pos, roll = self.detect_hand_gesture(landmarks)
                    self.process_actions(gesture, pos, roll, image)

                    if gesture != "NONE":
                        h, w, _ = image.shape
                        if pos:
                            cx, cy = int(pos.x * w), int(pos.y * h)
                        else:
                            cx, cy = int(landmarks[0].x * w), int(landmarks[0].y * h)

                        self.draw_glass_panel(image, (cx + 20, cy - 40), (cx + 160, cy - 10), C_GLASS_DARK)
                        self.draw_neon_text(image, gesture, (cx + 25, cy - 18), 0.5, C_MAGENTA_NEON)

                        CONNECTIONS = [
                            (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
                            (5,9), (9,10), (10,11), (11,12), (9,13), (13,14), (14,15), (15,16),
                            (13,17), (17,18), (18,19), (19,20), (0,17)
                        ]
                        for s, e in CONNECTIONS:
                            x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
                            x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
                            cv2.line(image, (x1, y1), (x2, y2), C_CYAN_DIM, 1)

                # --- UI LAYERS ---
                self.draw_hud_header(image, fps)
                self.draw_hud_sidebar(image, yaw_stable, pitch_stable)
                self.draw_status_overlay(image)

                cv2.imshow('EyeOS v2.4 [CYBERPUNK]', image)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord('c'):
                    self.calibration_mode = not self.calibration_mode
                    self.calibration_step = 0
                    self.log_action("CALIBRATION MODE")
                elif key == 32 and self.calibration_mode:
                    self.calibrate_step(yaw_stable, pitch_stable)
                elif key == ord('m'):
                    self.mouse_paused = not self.mouse_paused
                    self.log_action(f"MOUSE {'LOCKED' if self.mouse_paused else 'UNLOCKED'}")

        except KeyboardInterrupt:
            print("\n[EXIT] KeyboardInterrupt reçu, arrêt propre...")
        finally:
            self.shutdown()

if __name__ == "__main__":
    controller = None
    try:
        controller = EyeGestureController()
        controller.run()
    except Exception as e:
        print(f"Erreur Critique: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if controller is not None:
            try:
                controller.shutdown()
            except Exception:
                pass
