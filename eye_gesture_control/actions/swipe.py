class SwipeHandler:
    """Gestionnaire des gestes de swipe""" 
    def __init__(self, worker, log_callback):
        self.worker = worker
        self.log_callback = log_callback

    def handle(self, swipe_direction):
        """Gestion des gestes de swipe"""
        if swipe_direction == "RIGHT": # si le swipe est a droite
            self.worker.add_action("hotkey", ('ctrl', 'pagedown')) # on envoie la touche ctrl+pagedown
            self.log_callback("Tab Next >>") # on envoie le log
        else: # si le swipe est a gauche
            self.worker.add_action("hotkey", ('ctrl', 'pageup')) # on envoie la touche ctrl+pageup
            self.log_callback("<< Tab Prev") # on envoie le log

