import json
import os
import time
from screeninfo import get_monitors
from config import CONFIG_FILE

class CalibrationManager:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
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

    def log(self, message):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(f"[CALIBRATION] {message}")

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
            self.log(f"CALIB SAVED: {screen_name}")
            self.calibration_step += 1
            time.sleep(0.5)
        
        if self.calibration_step >= len(self.calibration_keys):
            self.calibration_mode = False
            self.save_calibration()
            self.log("CALIBRATION COMPLETE")

    def get_active_monitor(self, screen_name):
        if screen_name in self.screen_centers:
            idx = self.screen_centers[screen_name].get("monitor_idx", 0)
            if idx < len(self.monitors):
                return self.monitors[idx]
        return self.monitors[0] if self.monitors else None
