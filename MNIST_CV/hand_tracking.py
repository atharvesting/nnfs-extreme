import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

class HandTracker:

    def __init__(self, model_path="MNIST_CV/model/hand_landmarker.task", num_hands=2):
        self.stroke_points = []
        self.latest_result = None
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, 
                                               running_mode=vision.RunningMode.LIVE_STREAM,
                                               result_callback=self._result_callback,
                                               num_hands=num_hands
                                               )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles
        self.drawing_active = False

    def _result_callback(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.latest_result = result
        self.process_result(result, output_image)


    def detect(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        timestamp_ms = int(time.time() * 1000)
        self.landmarker.detect_async(mp_img, timestamp_ms)

    @staticmethod
    def chaikin_smooth(pts, iterations=5):
        if len(pts) <= 2:
            return pts
        for _ in range(iterations):
            Q = 0.75 * pts[:-1] + 0.25 * pts[1:]
            R = 0.25 * pts[:-1] + 0.75 * pts[1:]
            new_pts = np.empty((Q.shape[0] * 2, 2))
            new_pts[0::2] = Q
            new_pts[1::2] = R
            pts = np.vstack([pts[0], new_pts, pts[-1]])
        return pts

    def process_result(self, result, output_img: mp.Image):
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            
            # 3D distance helper
            def dist3d(l1, l2):
                return ((l1.x - l2.x)**2 + (l1.y - l2.y)**2 + (l1.z - l2.z)**2)**0.5
            
            wrist = landmarks[0]
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            
            # A finger is extended if its tip is further from the wrist than its PIP joint
            index_extended = dist3d(index_tip, wrist) > dist3d(index_pip, wrist)
            middle_extended = dist3d(middle_tip, wrist) > dist3d(middle_pip, wrist)
            
            if isinstance(output_img, mp.Image):
                h, w = output_img.height, output_img.width
            else:
                h, w, _ = output_img.shape
                
            index_coord = (int(index_tip.x * w), int(index_tip.y * h))
            
            if index_extended and not middle_extended:
                if self.drawing_active == False:
                    if dist3d(index_coord, self.stroke_points[-1][-1]) > 5:
                        self.stroke_points.append([index_coord])
                        self.drawing_active = True
                else:
                    self.stroke_points[-1].append(index_coord)
            else:
                self.drawing_active = False

    def bounding_boxes(self):
        boxes = []
        for digit in self.stroke_points:
            if len(digit) < 20:
                continue

            leftmost = min(digit, key=lambda x: x[0])[0]
            rightmost = max(digit, key=lambda x: x[0])[0]
            topmost = max(digit, key=lambda x: x[1])[1]
            bottommost = min(digit, key=lambda x: x[1])[1]

            x_ratio = rightmost - leftmost
            y_ratio = topmost - bottommost

            extend_factor = 0.2

            leftmost -= extend_factor * x_ratio
            rightmost += extend_factor * x_ratio
            topmost += extend_factor * y_ratio
            bottommost -= extend_factor * y_ratio

            boxes.append(
                (
                    # Top left point
                    (int(leftmost), int(topmost)),
                    # Bottom right point
                    (int(rightmost), int(bottommost))
                )
            )
        
        return boxes

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
        for stroke in self.stroke_points:
            if len(stroke) == 1:
                cv.circle(img, stroke[0], 2, (0, 255, 0), -1)
            elif len(stroke) > 1:
                pts = np.array(stroke, dtype=np.float32)
                pts = self.chaikin_smooth(pts, iterations=3)
                pts = pts.astype(np.int32).reshape((-1, 1, 2))
                cv.polylines(img, [pts], isClosed=False, color=(0, 255, 0), thickness=5)

        boxes = self.bounding_boxes()
        for pairs in boxes:
            cv.rectangle(img, pairs[0], pairs[1], color=(255, 0, 0), thickness=2)

        return img

    def clear_sketch(self):
        self.stroke_points = []