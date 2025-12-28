import time
import math
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import pygetwindow as gw

from config import *
from utils import Stabilizer
from workers import ActionWorker
from camera import CameraManager
from calibration import CalibrationManager
from head_tracking import HeadTracker
from hand_tracking import HandTracker
from ui import UIManager

class EyeGestureController:
    def __init__(self):
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # --- Modules ---
        self.worker = ActionWorker()
        self.worker.start()
        
        self.camera = CameraManager()
        self.calibration = CalibrationManager(log_callback=self.log_action)
        self.head_tracker = HeadTracker()
        self.hand_tracker = HandTracker()
        self.ui = UIManager()
        
        # --- État Système ---
        self.current_screen = "INCONNU"
        self.active_monitor = self.calibration.monitors[0] if self.calibration.monitors else None
        
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
        
        self.last_face_detected_time = time.time()
        self.is_afk = False
        self.mouse_paused = False

        print(f"Système initialisé. {len(self.calibration.monitors)} moniteurs détectés.")

    def log_action(self, message):
        print(f"[ACTION] {message}")
        self.last_log_message = message
        self.last_log_time = time.time()

    def determine_screen(self, yaw, pitch):
        best_screen = self.current_screen
        min_dist = float('inf')
        hysteresis_bonus = 5.0 

        for screen_name, data in self.calibration.screen_centers.items():
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
        self.active_monitor = self.calibration.get_active_monitor(screen_name)

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

    def process_actions(self, gesture, position, roll_angle, landmarks, image=None):
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
                swipe = self.hand_tracker.detect_swipe(position)
                if swipe:
                    if swipe == "RIGHT":
                        self.worker.add_action("hotkey", ('ctrl', 'pagedown'))
                        self.log_action("Tab Next >>")
                    else:
                        self.worker.add_action("hotkey", ('ctrl', 'pageup'))
                        self.log_action("<< Tab Prev")
                return

        speed = 0
        if gesture == "THREE_FINGERS" and position:
            DEADZONE = 15
            if roll_angle > DEADZONE:
                intensity = (roll_angle - DEADZONE) / 5
                speed = -int(SCROLL_SPEED * intensity)
            elif roll_angle < -DEADZONE:
                intensity = (abs(roll_angle) - DEADZONE) / 5
                speed = int(SCROLL_SPEED * intensity)
            
            if speed != 0:
                self.worker.add_action("scroll", (speed,))
        
        # Feedback visuel pour le scroll (géré dans UI maintenant, mais on passe speed)
        if image is not None and gesture == "THREE_FINGERS":
             self.ui.draw_gesture_feedback(image, gesture, position, landmarks, roll_angle, speed)
             return # On return ici car le draw_gesture_feedback est appelé plus bas pour les autres gestes aussi ? 
                    # Non, l'original avait un return.
        
        # ATTENTION: L'original faisait un return dans le bloc THREE_FINGERS.
        # Je dois m'assurer que UI est appelé.
        # Dans l'original:
        # if gesture == "THREE_FINGERS": ... draw ... return
        
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

    def shutdown(self):
        """Libère proprement toutes les ressources."""
        if hasattr(self, "worker") and self.worker:
            self.worker.stop()
        
        if hasattr(self, "camera") and self.camera:
            self.camera.release()

        cv2.destroyAllWindows()

        if hasattr(self, "head_tracker") and self.head_tracker:
            self.head_tracker.close()

        if hasattr(self, "hand_tracker") and self.hand_tracker:
            self.hand_tracker.close()

    def run(self):
        prev_time = time.time()

        try:
            while True:
                success, image = self.camera.read()
                if not success:
                    break

                self.frame_count += 1

                # FPS Calculation
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30
                prev_time = curr_time

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                )

                # --- Head Tracking ---
                try:
                    face_result = self.head_tracker.detect(mp_image)
                except KeyboardInterrupt:
                    raise
                
                yaw_stable, pitch_stable = 0, 0

                if face_result and face_result.face_landmarks:
                    self.last_face_detected_time = time.time()
                    if self.is_afk:
                        self.log_action("USER IDENTIFIED")
                        self.trigger_play_pause()
                        self.is_afk = False

                    landmarks = face_result.face_landmarks[0]
                    self.ui.draw_face_hud(image, landmarks, self.is_afk, self.calibration.calibration_mode)

                    raw_yaw, raw_pitch = self.head_tracker.get_head_pose(image.shape, landmarks)
                    yaw_stable = self.yaw_stabilizer.update(raw_yaw)
                    pitch_stable = self.pitch_stabilizer.update(raw_pitch)

                    if not self.calibration.calibration_mode:
                        new_screen = self.determine_screen(yaw_stable, pitch_stable)
                        if new_screen != self.current_screen:
                            self.current_screen = new_screen
                            self.update_active_monitor(new_screen)

                            # Reset stabilizers
                            self.mouse_stabilizer_x.value = None
                            self.mouse_stabilizer_y.value = None

                            # Reset lock poing
                            self.fist_state_locked = False
                            self.fist_release_start_time = 0

                else:
                    if not self.is_afk and (time.time() - self.last_face_detected_time > AFK_TIMEOUT):
                        self.log_action("NO USER - LOCKING")
                        self.trigger_play_pause()
                        self.is_afk = True

                # --- Hand Tracking ---
                try:
                    hand_result = self.hand_tracker.detect(mp_image)
                except KeyboardInterrupt:
                    raise

                if hand_result and hand_result.hand_landmarks and not self.calibration.calibration_mode:
                    landmarks = hand_result.hand_landmarks[0]
                    gesture, pos, roll = self.hand_tracker.detect_hand_gesture(landmarks)
                    
                    # Process Actions
                    self.process_actions(gesture, pos, roll, landmarks, image)
                    
                    # Draw Gesture Feedback (si pas déjà fait dans process_actions pour THREE_FINGERS)
                    # Note: process_actions appelle draw pour THREE_FINGERS.
                    # Pour les autres, on dessine ici.
                    if gesture != "THREE_FINGERS":
                        self.ui.draw_gesture_feedback(image, gesture, pos, landmarks, roll)

                # --- UI LAYERS ---
                self.ui.draw_hud_header(image, fps)
                self.ui.draw_hud_sidebar(image, yaw_stable, pitch_stable, self.last_log_message, self.start_time)
                self.ui.draw_status_overlay(image, self.current_screen, self.calibration.calibration_mode, self.is_afk, self.hand_tracker.last_swipe_display_time, self.hand_tracker.last_swipe_direction)

                cv2.imshow('EyeOS v2.4 [CYBERPUNK]', image)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord('c'):
                    self.calibration.calibration_mode = not self.calibration.calibration_mode
                    self.calibration.calibration_step = 0
                    self.log_action("CALIBRATION MODE")
                elif key == 32 and self.calibration.calibration_mode:
                    self.calibration.calibrate_step(yaw_stable, pitch_stable)
                elif key == ord('m'):
                    self.mouse_paused = not self.mouse_paused
                    self.log_action(f"MOUSE {'LOCKED' if self.mouse_paused else 'UNLOCKED'}")

        except KeyboardInterrupt:
            print("\n[EXIT] KeyboardInterrupt reçu, arrêt propre...")
        finally:
            self.shutdown()
