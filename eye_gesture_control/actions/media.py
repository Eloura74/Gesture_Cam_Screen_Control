import time
import pygetwindow as gw
import pyautogui
from config import PAUSE_COOLDOWN

class MediaHandler:
    def __init__(self, worker, log_callback):
        self.worker = worker
        self.log_callback = log_callback
        self.last_toggle_time = 0
        self.fist_state_locked = False
        self.fist_release_start_time = 0

    def trigger_play_pause_logic(self, active_monitor):
        """Active la fenêtre sous le curseur et envoie Espace (Version Robuste)"""
        if not active_monitor: return
        center_x = active_monitor.x + active_monitor.width // 2
        center_y = active_monitor.y + active_monitor.height // 2
        
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

    def trigger_play_pause(self, active_monitor):
        # Le cooldown est géré par la logique "fist_state_locked" mais on garde une sécu
        if time.time() - self.last_toggle_time < PAUSE_COOLDOWN:
            return
        
        self.log_callback("MEDIA TOGGLE (PAUSE/PLAY)")
        self.worker.add_action("custom", (self.trigger_play_pause_logic, [active_monitor]))
        self.last_toggle_time = time.time()

    def handle_fist(self, gesture, active_monitor):
        current_time = time.time()
        
        if gesture == "FIST":
            self.fist_release_start_time = 0 # Annule tout timer de déverrouillage
            
            # On déclenche seulement si non verrouillé ET cooldown passé
            if not self.fist_state_locked and (current_time - self.last_toggle_time > PAUSE_COOLDOWN):
                self.trigger_play_pause(active_monitor)
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

    def reset_lock(self):
        self.fist_state_locked = False
        self.fist_release_start_time = 0
