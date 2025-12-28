import cv2
import numpy as np

class CameraManager:
    def __init__(self, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.vignette_map = None
        self._generate_vignette_mask(width, height)

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

    def read(self):
        if not self.cap.isOpened():
            return False, None
        
        success, image = self.cap.read()
        if not success:
            return False, None

        image = cv2.flip(image, 1)

        # --- VIGNETTAGE OPTIMISÉ ---
        if self.vignette_map is None or image.shape[:2] != self.vignette_map.shape[:2]:
            h, w = image.shape[:2]
            self._generate_vignette_mask(w, h)

        if self.vignette_map is not None:
            image = (image * self.vignette_map).astype(np.uint8)
            
        return True, image

    def release(self):
        if self.cap:
            self.cap.release()
