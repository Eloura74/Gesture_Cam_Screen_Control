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

def check_and_download_models():
    import os
    import urllib.request
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    models = {
        "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    }
    for filename, url in models.items():
        target_path = os.path.join(base_path, filename)
        if not os.path.exists(target_path):
            try:
                urllib.request.urlretrieve(url, target_path)
                print(f"Modèle téléchargé: {target_path}")
            except Exception as e:
                print(f"Erreur téléchargement {filename}: {e}")
                raise
