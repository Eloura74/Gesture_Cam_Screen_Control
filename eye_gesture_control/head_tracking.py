import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import check_and_download_models

class HeadTracker:
    def __init__(self):
        check_and_download_models()
        
        cwd = os.path.dirname(os.path.abspath(__file__))
        face_model_path = os.path.join(cwd, 'face_landmarker.task')

        base_options_face = python.BaseOptions(model_asset_path=face_model_path)
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options_face)

    def detect(self, mp_image):
        try:
            return self.face_landmarker.detect(mp_image)
        except Exception as e:
            print(f"[FaceLandmarker] error: {e}")
            return None

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

    def close(self):
        if self.face_landmarker:
            self.face_landmarker.close()
