import cv2
import time
import numpy as np
import mediapipe as mp  # type: ignore[import]

from mediapipe.tasks.python import BaseOptions  # type: ignore[import]
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode  # type: ignore[import]
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizer, GestureRecognizerOptions, GestureRecognizerResult  # type: ignore[import]
from mediapipe.python.solutions import drawing_utils, hands, drawing_styles  # type: ignore[import]
from mediapipe.framework.formats import landmark_pb2  # type: ignore[import]

from config_ import HGDConfig


def get_mediapipe_model(config: HGDConfig) -> GestureRecognizer:
    """Get mediapipe model for gesture recognition (model is selected through config)."""

    # Setup mediapipe hand landmarking from:
    # - https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    # - https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt
    options = BaseOptions(model_asset_path=config.consts.model_asset_path)
    options = GestureRecognizerOptions(
        base_options=options,
        num_hands=1,
        running_mode=VisionTaskRunningMode.VIDEO,
    )
    # detector = vision.HandLandmarker.create_from_options(options)
    detector = GestureRecognizer.create_from_options(options)
    return detector


def draw_landmarks_on_image(
    rgb_image: np.ndarray,
    detection_result: GestureRecognizerResult,
    config: HGDConfig,
) -> np.ndarray:
    """Render hand landmarks (tip of fingers, knuckles, etc.) on the image.

    This code is adapted from Google's
    [MediaPipe Hand Gesture Recognizer](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt) example.
    """
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                )
                for landmark in hand_landmarks
            ]
        )
        drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            hands.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - config.annotation_config.margin

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            config.annotation_config.font_size,
            config.annotation_config.hand_text_colour,
            config.annotation_config.font_thickness,
            cv2.LINE_AA,
        )

    height, width, _ = annotated_image.shape
    cv2.rectangle(
        annotated_image,
        config.mouse_movement_config.mouse_deadzone.start_coordinates(width, height),
        config.mouse_movement_config.mouse_deadzone.end_coordinates(width, height),
        # (int(width / 2 - DEADZONE * width), int(height / 2 - DEADZONE * height)),
        # (int(width / 2 + DEADZONE * width), int(height / 2 + DEADZONE * height)),
        config.annotation_config.box_colour,
        1,
    )
    cv2.putText(
        annotated_image,
        f"DEADZONE",
        config.mouse_movement_config.mouse_deadzone.start_coordinates(width, height),
        # (int(width / 2 - DEADZONE * width), int(height / 2 - DEADZONE * height)),
        cv2.FONT_HERSHEY_DUPLEX,
        config.annotation_config.font_size,
        config.annotation_config.box_colour,
        config.annotation_config.font_thickness,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        annotated_image,
        config.scroll_config.scroll_zone.start_coordinates(width, height),
        config.scroll_config.scroll_zone.end_coordinates(width, height),
        # (int(width / 2), int(height / 2 - DEADZONE * height)),
        # (int(width / 2 + DEADZONE * width), int(height / 2 + DEADZONE * height)),
        config.annotation_config.box_colour,
        1,
    )
    cv2.putText(
        annotated_image,
        f"SCROLLING ZONE",
        config.scroll_config.scroll_zone.start_coordinates(width, height),
        # (int(width / 2), int(height / 2 - DEADZONE * height)),
        cv2.FONT_HERSHEY_DUPLEX,
        config.annotation_config.font_size,
        config.annotation_config.box_colour,
        config.annotation_config.font_thickness,
        cv2.LINE_AA,
    )

    return annotated_image


def predict(
    frame: cv2.typing.MatLike,
    detector: GestureRecognizer,
    config: HGDConfig,
) -> tuple[GestureRecognizerResult, np.ndarray, bool]:
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # detection_result = detector.detect_for_video(
    detection_result = detector.recognize_for_video(
        mp_frame,
        timestamp_ms=int(time.time() * 1000),
    )
    annotated_image = draw_landmarks_on_image(mp_frame.numpy_view(), detection_result, config)
    hand_detected = bool(detection_result.hand_landmarks)
    return detection_result, annotated_image, hand_detected
