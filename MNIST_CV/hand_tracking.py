import cv2 as cv
import numpy as np
from numpy.typing import NDArray
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

class HandTracker:

    def __init__(self, model_path="MNIST_CV/model/hand_landmarker.task", num_hands=2):
        self.stroke_points: list[list[tuple]] = []
        self.latest_result: vision.HandLandmarkerResult | None = None
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
        self.drawing_active: bool = False

    def _result_callback(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
        self.latest_result = result
        self.process_result(result, output_image)

    def detect(self, frame: NDArray) -> None:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_frame: mp.Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms: int = int(time.time() * 1000)
        self.landmarker.detect_async(mp_frame, timestamp_ms)

    @staticmethod
    def chaikin_smooth(pts, iterations=5) -> NDArray:
        if len(pts) <= 20:
            return pts
        for _ in range(iterations):
            Q = 0.75 * pts[:-1] + 0.25 * pts[1:]
            R = 0.25 * pts[:-1] + 0.75 * pts[1:]
            new_pts = np.empty((Q.shape[0] * 2, 2))
            new_pts[0::2] = Q
            new_pts[1::2] = R
            pts = np.vstack([pts[0], new_pts, pts[-1]])
        return pts

    @staticmethod
    def dist3d(l1, l2) -> float:
        return ((l1.x - l2.x)**2 + (l1.y - l2.y)**2 + (l1.z - l2.z)**2)**0.5

    def process_result(self, result: vision.HandLandmarkerResult, output_frame: mp.Image) -> None:
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            
            wrist = landmarks[0]
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            
            # A finger is extended if its tip is further from the wrist than its PIP joint
            index_extended: bool = self.dist3d(index_tip, wrist) > self.dist3d(index_pip, wrist)
            middle_extended: bool = self.dist3d(middle_tip, wrist) > self.dist3d(middle_pip, wrist)
            
            if isinstance(output_frame, mp.Image):
                h, w = output_frame.height, output_frame.width
            else:
                h, w, _ = output_frame.shape
                
            index_coord: tuple[int, int] = (int(index_tip.x * w), int(index_tip.y * h))
            
            if index_extended and not middle_extended:
                if self.drawing_active == False:
                    self.stroke_points.append([index_coord])
                    self.drawing_active = True
                else:
                    self.stroke_points[-1].append(index_coord)
            else:
                self.drawing_active = False

    def bounding_boxes(self) -> list[tuple[tuple[int, int, int]]]:
        boxes: list[tuple[tuple[int, int, int]]] = []
        for digit in self.stroke_points:
            if len(digit) < 30:
                continue

            leftmost: int = min(digit, key=lambda x: x[0])[0]
            rightmost: int = max(digit, key=lambda x: x[0])[0]
            topmost: int = max(digit, key=lambda x: x[1])[1]
            bottommost: int = min(digit, key=lambda x: x[1])[1]

            delta_x: int = rightmost - leftmost
            delta_y: int = topmost - bottommost
    
            extend_factor: float = 0.35
            topmost     += delta_y * extend_factor
            bottommost  -= delta_y * extend_factor

            new_delta_y: int = topmost - bottommost

            p: float = (new_delta_y - delta_x) / 2
            leftmost -= p
            rightmost += p

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
    def draw_landmarks_on_image(self, rgb_image: mp.Image, detection_result: vision.HandLandmarkerResult) -> mp.Image:
        hand_landmarks_list: list[list] = detection_result.hand_landmarks
        annotated_image: mp.Image = np.copy(rgb_image)

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

    def sketch(self, frame, boxes) -> NDArray:
        for stroke in self.stroke_points:
            if len(stroke) == 1:
                cv.circle(frame, stroke[0], 2, (0, 255, 0), -1)
            elif len(stroke) > 1:
                pts: NDArray = np.array(stroke, dtype=np.float32)
                pts = self.chaikin_smooth(pts, iterations=3)
                pts = pts.astype(np.int32).reshape((-1, 1, 2))
                cv.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=15)

        for pairs in boxes:
            cv.rectangle(frame, pairs[0], pairs[1], color=(255, 0, 0), thickness=1)

        return frame

    def get_box_images(self, frame, boxes) -> list[NDArray]:
        image_arrays: list[NDArray] = []
        for pairs in boxes:
            image_arrays.append(frame
                [
                    pairs[1][1]:pairs[0][1],
                    pairs[0][0]:pairs[1][0]
                ]
            )

        return image_arrays

    def clear_sketch(self) -> None:
        self.stroke_points.clear()