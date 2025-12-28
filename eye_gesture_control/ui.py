import cv2
import numpy as np
import time
import math
from config import *

class UIManager:
    def __init__(self):
        pass

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

    def draw_hud_sidebar(self, image, yaw, pitch, last_log_message, start_time):
        h, w, _ = image.shape
        # Panneau Gauche: Logs
        self.draw_glass_panel(image, (10, h-100), (300, h-10), C_GLASS_DARK, 0.5)
        self.draw_neon_text(image, f">> {last_log_message}", (20, h-50), 0.45, C_WHITE)
        # Timer
        uptime = int(time.time() - start_time)
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

    def draw_face_hud(self, image, landmarks, is_afk, calibration_mode):
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
        color = C_ALERT_RED if is_afk else C_CYAN_BRIGHT
        if calibration_mode: color = C_MAGENTA_NEON

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

    def draw_status_overlay(self, image, current_screen, calibration_mode, is_afk, last_swipe_time, last_swipe_direction):
        """Affiche les écrans/modes au centre si changement"""
        h, w, _ = image.shape
        cx, cy = w//2, h//2
        
        # Affichage Ecran Actif en Haut
        screen_label = current_screen.replace("ECRAN_", "").replace("_", " ")
        self.draw_glass_panel(image, (cx-100, 50), (cx+100, 80), C_GLASS_DARK, 0.8)
        self.draw_neon_text(image, screen_label, (cx-90, 72), 0.6, C_BLUE_TECH, 1)

        # Mode Calibration
        if calibration_mode:
            cv2.rectangle(image, (0,0), (w,h), C_MAGENTA_DEEP, 20) # Bordure massive
            self.draw_neon_text(image, "CALIBRATION ENGAGED", (cx-150, cy), 1.0, C_MAGENTA_NEON, 2)
            self.draw_neon_text(image, "LOOK AT TARGET & PRESS SPACE", (cx-180, cy+40), 0.6, C_WHITE, 1)

        # Mode AFK
        if is_afk:
            overlay = image.copy()
            cv2.rectangle(overlay, (0,0), (w,h), (0,0,50), -1) # Fond rouge sombre
            cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
            self.draw_neon_text(image, "SYSTEM LOCKED / AFK", (cx-140, cy), 0.8, C_ALERT_RED, 2)
            
        # Feedback Swipe
        if time.time() - last_swipe_time < 1.0 and last_swipe_direction:
            text = f"SWIPE {last_swipe_direction} >>"
            self.draw_glass_panel(image, (cx-100, cy+100), (cx+100, cy+140), C_GLASS_DARK, 0.8)
            self.draw_neon_text(image, text, (cx-80, cy+128), 0.7, C_GREEN_DATA)

    def draw_gesture_feedback(self, image, gesture, pos, landmarks, roll_angle, scroll_speed=0):
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
        
        if gesture == "THREE_FINGERS" and pos:
             # FEEDBACK VISUEL RESTAURÉ (Joystick)
            h, w, _ = image.shape
            cx, cy = int(pos.x * w), int(pos.y * h)
            color_joy = (0, 255, 255) # Jaune/Cyan
            DEADZONE = 15
            if abs(roll_angle) > DEADZONE: color_joy = (0, 255, 0) # Vert quand actif
            
            # Cercle Extérieur
            cv2.circle(image, (cx, cy), 45, color_joy, 2)
            
            # Ligne Directionnelle
            rad = math.radians(roll_angle - 90)
            ex = int(cx + 45 * math.cos(rad))
            ey = int(cy + 45 * math.sin(rad))
            cv2.line(image, (cx, cy), (ex, ey), color_joy, 2)
            
            # Texte
            cv2.putText(image, f"SCROLL {scroll_speed}", (cx-35, cy-55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_joy, 1)
