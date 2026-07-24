import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

class HandTracker:

    def __init__(self, model_path="MNIST_CV/model/hand_landmarker.task", num_hands=2):
        self.stroke_points = []
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, 
                                               running_mode=vision.RunningMode.IMAGE,
                                            #    result_callback=print_result,
                                               num_hands=num_hands
                                               )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles


    def detect(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        timestamp_ms = int(time.time() * 1000)
        return self.landmarker.detect(mp_img)

    @staticmethod
    def point_distance(p1: tuple, p2: tuple) -> float:
        vec = (p2[0] - p1[0], p2[1] - p1[1])
        return ( (vec[0] ** 2) + vec[1] ** 2 ) ** 0.5

    def process_result(self, result, output_img: mp.Image):
        if result.hand_landmarks:
            index_obj = result.hand_landmarks[0][8]
            thumb_obj = result.hand_landmarks[0][4]
            h, w, _ = output_img.shape
            norm = lambda obj : (int(obj.x * w), int(obj.y * h))
            index_coord = norm(index_obj)
            thumb_coord = norm(thumb_obj)
            if self.point_distance(index_coord, thumb_coord) < 30:
                self.stroke_points.append(index_coord)

    # Source:  https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Draw the hand landmarks.
            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        return annotated_image

    def sketch(self, img):
        points_array = np.array(self.stroke_points, dtype=np.int32).reshape((-1, 1, 2))
        cv.polylines(img, [points_array], isClosed=False, color=(0, 255, 0), thickness=5)
        return img

    def clear_sketch(self):
        self.stroke_points = []