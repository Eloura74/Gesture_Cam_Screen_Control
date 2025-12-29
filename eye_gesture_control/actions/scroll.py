from config import SCROLL_SPEED

class ScrollHandler:
    def __init__(self, worker): 
        self.worker = worker 
# fonction de gestion du scroll
    def handle(self, roll_angle):
        speed = 0 # vitesse de scroll
        DEADZONE = 15 # zone de deadzone
        if roll_angle > DEADZONE: # si le roll est superieur a la deadzone
            intensity = (roll_angle - DEADZONE) / 5 # on calcule l'intensite
            speed = -int(SCROLL_SPEED * intensity) # on calcule la vitesse
        elif roll_angle < -DEADZONE: # si le roll est inferieur a la deadzone
            intensity = (abs(roll_angle) - DEADZONE) / 5 # on calcule l'intensite
            speed = int(SCROLL_SPEED * intensity) # on calcule la vitesse
        
        if speed != 0: # si la vitesse est differente de 0
            self.worker.add_action("scroll", (speed,)) # on envoie la vitesse
        return speed
