import numpy as np
from utils import Stabilizer
from config import MOUSE_SMOOTHING

class MouseHandler:
    def __init__(self, worker):
        self.worker = worker
        self.mouse_stabilizer_x = Stabilizer(MOUSE_SMOOTHING)
        self.mouse_stabilizer_y = Stabilizer(MOUSE_SMOOTHING)
        self.paused = False

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused

    def reset_stabilizers(self):
        self.mouse_stabilizer_x.value = None
        self.mouse_stabilizer_y.value = None

    def handle(self, position, active_monitor):
        if self.paused or not active_monitor:
            return

        # Clamping pour éviter l'extrapolation hors bornes
        px = min(max(position.x, 0.1), 0.9)
        py = min(max(position.y, 0.1), 0.9)
        
        screen_x_rel = np.interp(px, [0.1, 0.9], [0, active_monitor.width]) # on calcule la position relative de la souris
        screen_y_rel = np.interp(py, [0.1, 0.9], [0, active_monitor.height]) # on calcule la position relative de la souris
        
        target_x = active_monitor.x + screen_x_rel # on calcule la position absolute de la souris
        target_y = active_monitor.y + screen_y_rel # on calcule la position absolute de la souris
        
        target_x = max(active_monitor.x, min(target_x, active_monitor.x + active_monitor.width - 1)) # on limite la position de la souris
        target_y = max(active_monitor.y, min(target_y, active_monitor.y + active_monitor.height - 1)) # on limite la position de la souris

        # Application du lissage souris (Stabilizer)
        smooth_x = self.mouse_stabilizer_x.update(target_x) # on applique le lissage souris
        smooth_y = self.mouse_stabilizer_y.update(target_y) # on applique le lissage souris

        # Envoi INT pour éviter les problèmes de drivers
        self.worker.add_action("move", (int(smooth_x), int(smooth_y))) # on envoie la position de la souris
