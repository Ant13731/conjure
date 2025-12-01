import cv2
import time
import numpy as np
import mediapipe as mp  # type: ignore[import]

from mediapipe.tasks.python import BaseOptions  # type: ignore[import]
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode  # type: ignore[import]
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizer, GestureRecognizerOptions, GestureRecognizerResult  # type: ignore[import]
from mediapipe.python.solutions import drawing_utils, hands, drawing_styles  # type: ignore[import]
from mediapipe.framework.formats import landmark_pb2  # type: ignore[import]

from enum import StrEnum
from dataclasses import dataclass
from loguru import logger


class Gesture(StrEnum):
    unknown = "unknown"
    # closed_fist = auto()
    # open_palm = auto()
    # point_up = auto()
    # thumbs_down = auto()
    # thumbs_up = auto()
    # victory = auto()
    # love = auto()
    fist = "fist"
    ok = "ok"
    one = "one"
    palm = "palm"
    peace = "peace"
    peace_inverted = "peace_inverted"
    rock = "rock"
    stop = "stop"
    stop_inverted = "stop_inverted"
    two_up = "two_up"
    two_up_inverted = "two_up_inverted"

    @staticmethod
    def from_str(label: str) -> "Gesture":
        try:
            return Gesture(label)
        except ValueError:
            return Gesture.unknown


class Handedness(StrEnum):
    unknown = "unknown"
    left = "left"
    right = "right"

    @staticmethod
    def from_str(label: str) -> "Handedness":
        try:
            return Handedness(str.lower(label))
        except ValueError:
            return Handedness.unknown


# def depth_at_normalized_interpolated(depth_map, x_norm, y_norm):
#     H, W = depth_map.shape

#     # Convert normalized → continuous pixel coordinates
#     x = x_norm * (W - 1)
#     y = y_norm * (H - 1)

#     # Integer pixel locations
#     x0 = int(np.floor(x))
#     x1 = min(x0 + 1, W - 1)
#     y0 = int(np.floor(y))
#     y1 = min(y0 + 1, H - 1)

#     # Fractional part
#     dx = x - x0
#     dy = y - y0

#     # Fetch the four neighbors
#     Q11 = depth_map[y0, x0]
#     Q21 = depth_map[y0, x1]
#     Q12 = depth_map[y1, x0]
#     Q22 = depth_map[y1, x1]

#     # Bilinear interpolation formula
#     top = Q11 * (1 - dx) + Q21 * dx
#     bottom = Q12 * (1 - dx) + Q22 * dx
#     value = top * (1 - dy) + bottom * dy

#     return value


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float = 0.0


def min_depth_in_surrounding_area(depth_map, x_norm, y_norm, area_size=3):
    H, W = depth_map.shape

    # Convert normalized → continuous pixel coordinates
    x = x_norm * (W - 1)
    y = y_norm * (H - 1)

    # Integer pixel locations
    x_center = int(np.round(x))
    y_center = int(np.round(y))

    # Define the surrounding area
    x_start = max(x_center - area_size // 2, 0)
    x_end = min(x_center + area_size // 2 + 1, W)
    y_start = max(y_center - area_size // 2, 0)
    y_end = min(y_center + area_size // 2 + 1, H)

    if x_center > W or x_center < 0 or y_center > H or y_center < 0:
        return 100

    if x_start >= x_end or y_start >= y_end:
        logger.warning("Surrounding area is out of bounds, returning center depth")
        return depth_map[y_center, x_center]

    # Extract the surrounding area
    surrounding_area = depth_map[y_start:y_end, x_start:x_end]

    # Return the minimum depth in the surrounding area
    return np.min(surrounding_area)


@dataclass
class GestureRecognizerCustomResult:
    hand_detected: bool
    gesture: Gesture
    gesture_confidence: float
    handedness: Handedness
    handedness_confidence: float
    landmarks: list[Landmark]

    @property
    def thumb_tip(self) -> Landmark:
        return self.landmarks[4]

    @property
    def index_finger_tip(self) -> Landmark:
        return self.landmarks[8]

    @property
    def middle_finger_tip(self) -> Landmark:
        return self.landmarks[12]

    @property
    def ring_finger_tip(self) -> Landmark:
        return self.landmarks[16]

    @property
    def pinky_finger_tip(self) -> Landmark:
        return self.landmarks[20]

    @property
    def wrist(self) -> Landmark:
        return self.landmarks[0]

    @classmethod
    def from_mediapipe_result(cls, result: GestureRecognizerResult, depth_map) -> "GestureRecognizerCustomResult":
        if not result.hand_landmarks:
            return cls(
                hand_detected=False,
                gesture=Gesture.unknown,
                gesture_confidence=0.0,
                handedness=Handedness.unknown,
                handedness_confidence=0.0,
                landmarks=[],
            )

        if not result.gestures or not result.gestures[0]:
            gesture = Gesture.unknown
            gesture_confidence = 0.0
        else:
            gesture = Gesture.from_str(result.gestures[0][0].category_name)
            gesture_confidence = result.gestures[0][0].score

        if not result.handedness or not result.handedness[0]:
            handedness = Handedness.unknown
            handedness_confidence = 0.0
        else:
            handedness = Handedness.from_str(result.handedness[0][0].category_name)
            handedness_confidence = result.handedness[0][0].score

        landmarks = []

        for landmark in result.hand_landmarks[0]:
            # assume that hands will be the closest thing to the camera
            # depth then becomes the min of surrounding pixels
            # dimensions = lambda center: (max(int(center * 480) - 1, 0), min(int(center * 480) + 1, 479))
            # z = np.min(depth_map[dimensions(landmark.x)[0] : dimensions(landmark.x)[1], dimensions(landmark.y)[0] : dimensions(landmark.y)[1]])

            landmarks.append(
                Landmark(
                    x=landmark.x,
                    y=landmark.y,
                    # z=depth_at_normalized_interpolated(depth_map, landmark.x, landmark.y),
                    z=min_depth_in_surrounding_area(depth_map, landmark.x, landmark.y),
                    visibility=landmark.visibility,
                ),
            )

        return cls(
            hand_detected=True,
            gesture=gesture,
            gesture_confidence=gesture_confidence,
            handedness=handedness,
            handedness_confidence=handedness_confidence,
            landmarks=landmarks,
        )


def get_mediapipe_model() -> GestureRecognizer:
    """Get mediapipe model for gesture recognition (model is selected through config)."""

    # Setup mediapipe hand landmarking from:
    # - https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    # - https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt
    # options = BaseOptions(model_asset_path="./src/gesture_recognizer.task")
    options = BaseOptions(model_asset_path="./src/trained_mediapipe_gesture_recognizer.task")
    options = GestureRecognizerOptions(
        base_options=options,
        num_hands=1,
        running_mode=VisionTaskRunningMode.VIDEO,
    )
    # detector = vision.HandLandmarker.create_from_options(options)
    detector = GestureRecognizer.create_from_options(options)
    return detector


def swap_handedness_for_display(handedness_category: str) -> str:
    """Swap handedness category for display (since the camera is flipped to mirror the user)."""
    if handedness_category == "Left":
        return "Right"
    elif handedness_category == "Right":
        return "Left"
    return handedness_category


DEPTH_THRESHOLD = 0.25
GREYED_OUT_ALPHA = 0.45
GLASS_PANE_SIZE = 400


def apply_glass_for_far_depth(image, depth):
    h, w = image.shape[:2]

    pane_mask = np.zeros((h, w), dtype=np.uint8)

    x1 = w // 2 - GLASS_PANE_SIZE // 2
    y1 = h // 2 - GLASS_PANE_SIZE // 2
    x2 = x1 + GLASS_PANE_SIZE
    y2 = y1 + GLASS_PANE_SIZE

    radius = 30

    # Draw rounded rect into mask
    cv2.rectangle(pane_mask, (x1 + radius, y1), (x2 - radius, y2), 255, -1)
    cv2.rectangle(pane_mask, (x1, y1 + radius), (x2, y2 - radius), 255, -1)
    cv2.circle(pane_mask, (x1 + radius, y1 + radius), radius, 255, -1)
    cv2.circle(pane_mask, (x2 - radius, y1 + radius), radius, 255, -1)
    cv2.circle(pane_mask, (x1 + radius, y2 - radius), radius, 255, -1)
    cv2.circle(pane_mask, (x2 - radius, y2 - radius), radius, 255, -1)

    depth_mask = depth > np.float16(DEPTH_THRESHOLD)
    mask = np.logical_and(pane_mask > 0, depth_mask)

    overlay = np.zeros_like(image)
    overlay[mask] = (200, 200, 200)

    result = cv2.addWeighted(overlay, GREYED_OUT_ALPHA, image, 1 - GREYED_OUT_ALPHA, 0)

    return result


def depth_to_alpha(z, max_depth=1.0, power=0.3):
    """
    Maps depth to brightness/opacity:
    - z <= threshold: behind glass → alpha=0
    - threshold < z < max_depth: gradually increase alpha
    - z >= max_depth: alpha=1
    """
    if z <= DEPTH_THRESHOLD:
        return 1.0

    if z >= max_depth:
        return 0.0

    # Normalize distance from threshold
    t = (z - DEPTH_THRESHOLD) / (max_depth - DEPTH_THRESHOLD)

    # Exponential falloff for drastic effect
    alpha = 1.0 - (t**power)

    return alpha


def draw_fingertip(image, landmark: Landmark):
    px = int(landmark.x * image.shape[1])
    py = int(landmark.y * image.shape[0])

    # Get normalized alpha / brightness
    alpha = depth_to_alpha(landmark.z)

    # Map alpha to color (from dark blue to bright blue, for example)
    bright_color = np.array([255, 200, 0], dtype=np.uint8)
    dark_color = np.array([0, 0, 0], dtype=np.uint8)

    # Interpolate
    color = (dark_color * (1 - alpha) + bright_color * alpha).astype(np.uint8)

    # Circle radius
    radius = 10

    cv2.circle(image, (px, py), radius, color.tolist(), -1)


def draw_circle(image, depth_map, x, y):
    px = max(min(int(x * image.shape[1]), 479), 0)
    py = max(min(int(y * image.shape[0]), 479), 0)

    # Get normalized alpha / brightness
    alpha = depth_to_alpha(depth_map[py, px])

    # Map alpha to color (from dark blue to bright blue, for example)
    bright_color = np.array([255, 255, 255], dtype=np.uint8)
    dark_color = np.array([0, 0, 0], dtype=np.uint8)

    # Interpolate
    color = (dark_color * (1 - alpha) + bright_color * alpha).astype(np.uint8)

    # Circle radius
    radius = 10

    cv2.circle(image, (px, py), radius, color.tolist(), -1)


def draw_landmarks_on_image(
    rgb_image: np.ndarray,
    depth_map: np.ndarray,
    detection_result: GestureRecognizerCustomResult,
    # config: HGDConfig,
) -> np.ndarray:
    """Render hand landmarks (tip of fingers, knuckles, etc.) on the image.

    This code is adapted from Google's
    [MediaPipe Hand Gesture Recognizer](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt) example.
    """

    # grey_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    # grey_image = cv2.cvtColor(grey_image, cv2.COLOR_GRAY2BGR)
    # greyed_image_overlay = cv2.addWeighted(grey_image, GREYED_OUT_ALPHA, rgb_image, 1 - GREYED_OUT_ALPHA, 0)

    annotated_image = np.copy(rgb_image)
    # annotated_image = apply_glass_for_far_depth(annotated_image, depth_map)
    # annotated_image = np.copy(annotated_image)
    # annotated_image[depth_map > DEPTH_THRESHOLD * 255] = greyed_image_overlay[depth_map > DEPTH_THRESHOLD * 255]

    if not detection_result.hand_detected:
        return annotated_image

    # hand_landmarks = detection_result.landmarks
    # handedness = detection_result.handedness
    # print("Index depth:", detection_result.index_finger_tip.z)
    # print("Index x,y:", detection_result.index_finger_tip.x, detection_result.index_finger_tip.y)
    for i in range(20):
        for j in range(20):
            draw_circle(annotated_image, depth_map, i / 20, j / 20)
    draw_fingertip(annotated_image, detection_result.index_finger_tip)
    draw_fingertip(annotated_image, detection_result.thumb_tip)
    draw_fingertip(annotated_image, detection_result.wrist)
    draw_fingertip(annotated_image, detection_result.middle_finger_tip)
    # SEPARATION = 0.01
    # draw_circle(annotated_image, depth_map, detection_result.wrist.x, detection_result.wrist.y)
    # draw_circle(annotated_image, depth_map, detection_result.wrist.x + SEPARATION, detection_result.wrist.y)
    # draw_circle(annotated_image, depth_map, detection_result.wrist.x + SEPARATION, detection_result.wrist.y + SEPARATION)
    # draw_circle(annotated_image, depth_map, detection_result.wrist.x, detection_result.wrist.y + SEPARATION)
    # draw_circle(annotated_image, depth_map, detection_result.wrist.x, detection_result.wrist.y - SEPARATION)
    # draw_circle(annotated_image, depth_map, detection_result.wrist.x - SEPARATION, detection_result.wrist.y - SEPARATION)
    # draw_circle(annotated_image, depth_map, detection_result.wrist.x - SEPARATION, detection_result.wrist.y)
    # # draw_fingertip(annotated_image, detection_result.index_finger_tip)  # config)
    # Loop through the detected hands to visualize.
    # for idx in range(len(hand_landmarks_list)):
    # hand_landmarks = hand_landmarks_list[idx]
    # handedness = handedness_list[idx]

    # Draw the hand landmarks.
    # hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()  # type: ignore
    # hand_landmarks_proto.landmark.extend(
    #     [
    #         landmark_pb2.NormalizedLandmark(  # type: ignore
    #             x=landmark.x,
    #             y=landmark.y,
    #             z=landmark.z,
    #         )
    #         for landmark in hand_landmarks
    #     ]
    # )
    # drawing_utils.draw_landmarks(
    #     annotated_image,
    #     hand_landmarks_proto,
    #     hands.HAND_CONNECTIONS,  # type: ignore
    #     drawing_styles.get_default_hand_landmarks_style(),
    #     drawing_styles.get_default_hand_connections_style(),
    # )
    # Get the top left corner of the detected hand's bounding box.
    # height, width, _ = annotated_image.shape
    # x_coordinates = [landmark.x for landmark in hand_landmarks]
    # y_coordinates = [landmark.y for landmark in hand_landmarks]
    # text_x = int(min(x_coordinates) * width)
    # text_y = int(min(y_coordinates) * height) - 10  # config.annotation_config.margin
    # # Draw handedness (left or right hand) on the image.
    # cv2.putText(
    #     annotated_image,
    #     f"{swap_handedness_for_display(handedness.value)}",
    #     (text_x, text_y),
    #     cv2.FONT_HERSHEY_DUPLEX,
    #     1,  # config.annotation_config.font_size,
    #     (88, 205, 54),  # config.annotation_config.hand_text_colour,
    #     1,  # config.annotation_config.font_thickness,
    #     cv2.LINE_AA,
    # )

    # height, width, _ = annotated_image.shape
    # cv2.rectangle(
    #     annotated_image,
    #     config.mouse_movement_config.mouse_deadzone.start_coordinates(width, height),
    #     config.mouse_movement_config.mouse_deadzone.end_coordinates(width, height),
    #     # (int(width / 2 - DEADZONE * width), int(height / 2 - DEADZONE * height)),
    #     # (int(width / 2 + DEADZONE * width), int(height / 2 + DEADZONE * height)),
    #     config.annotation_config.box_colour,
    #     1,
    # )
    # cv2.putText(
    #     annotated_image,
    #     f"DEADZONE",
    #     config.mouse_movement_config.mouse_deadzone.start_coordinates(width, height),
    #     # (int(width / 2 - DEADZONE * width), int(height / 2 - DEADZONE * height)),
    #     cv2.FONT_HERSHEY_DUPLEX,
    #     config.annotation_config.font_size,
    #     config.annotation_config.box_colour,
    #     config.annotation_config.font_thickness,
    #     cv2.LINE_AA,
    # )

    # cv2.rectangle(
    #     annotated_image,
    #     config.scroll_config.scroll_zone.start_coordinates(width, height),
    #     config.scroll_config.scroll_zone.end_coordinates(width, height),
    #     # (int(width / 2), int(height / 2 - DEADZONE * height)),
    #     # (int(width / 2 + DEADZONE * width), int(height / 2 + DEADZONE * height)),
    #     config.annotation_config.box_colour,
    #     1,
    # )
    # cv2.putText(
    #     annotated_image,
    #     f"SCROLL ZONE",
    #     config.scroll_config.scroll_zone.start_coordinates(width, height),
    #     # (int(width / 2), int(height / 2 - DEADZONE * height)),
    #     cv2.FONT_HERSHEY_DUPLEX,
    #     config.annotation_config.font_size,
    #     config.annotation_config.box_colour,
    #     config.annotation_config.font_thickness,
    #     cv2.LINE_AA,
    # )

    return annotated_image


def predict(
    frame: cv2.typing.MatLike,
    detector: GestureRecognizer,
    depth_map: np.ndarray,
    # config: HGDConfig,
) -> tuple[GestureRecognizerCustomResult, np.ndarray]:
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # detection_result = detector.detect_for_video(
    detection_result = detector.recognize_for_video(
        mp_frame,
        timestamp_ms=int(time.time() * 1000),
    )
    custom_result = GestureRecognizerCustomResult.from_mediapipe_result(detection_result, depth_map)
    annotated_image = draw_landmarks_on_image(
        mp_frame.numpy_view(),
        depth_map,
        custom_result,
    )  # config)
    return custom_result, annotated_image


# GestureRecognizerResult(
#     gestures=[[Category(index=-1, score=0.6564996242523193, display_name="", category_name="Victory")]],
#     handedness=[[Category(index=1, score=0.9799932241439819, display_name="Left", category_name="Left")]],
#     hand_landmarks=[
#         [
#             NormalizedLandmark(x=0.9132241010665894, y=0.9648247361183167, z=5.774457463303406e-07, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.8312482237815857, y=0.9533495306968689, z=-0.00748348468914628, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7744005918502808, y=0.9280455112457275, z=-0.021088367328047752, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.737042248249054, y=0.906495988368988, z=-0.03949720785021782, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7172938585281372, y=0.8922374844551086, z=-0.05689404159784317, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7701709866523743, y=0.8307397365570068, z=-0.017229324206709862, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.713687539100647, y=0.7734889388084412, z=-0.04154350236058235, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.6808316111564636, y=0.7386624217033386, z=-0.05729442834854126, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.6588999032974243, y=0.7076488733291626, z=-0.06804708391427994, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.8064779043197632, y=0.8198879957199097, z=-0.032520536333322525, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7603040337562561, y=0.7558884620666504, z=-0.06411638110876083, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7342799305915833, y=0.7110357284545898, z=-0.08763639628887177, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7181337475776672, y=0.673639178276062, z=-0.10151121765375137, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.839346170425415, y=0.8294984698295593, z=-0.04844033718109131, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7726945877075195, y=0.8143261075019836, z=-0.0852479413151741, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7365102171897888, y=0.8538879156112671, z=-0.09875386953353882, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7183957099914551, y=0.8905735611915588, z=-0.10018617659807205, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.8613342046737671, y=0.8592783212661743, z=-0.06431285291910172, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.8003712296485901, y=0.8695933222770691, z=-0.09199162572622299, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7605620622634888, y=0.8881784081459045, z=-0.09924615174531937, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7318859696388245, y=0.905154824256897, z=-0.09949765354394913, visibility=0.0, presence=0.0),
#         ]
#     ],
#     hand_world_landmarks=[
#         [
#             Landmark(x=0.05314017832279205, y=0.07341700792312622, z=0.03937985748052597, visibility=0.0, presence=0.0),
#             Landmark(x=0.016959959641098976, y=0.05913615971803665, z=0.04245372861623764, visibility=0.0, presence=0.0),
#             Landmark(x=-0.010125977918505669, y=0.046665437519550323, z=0.03173660486936569, visibility=0.0, presence=0.0),
#             Landmark(x=-0.03149949014186859, y=0.037893589586019516, z=0.0010267328470945358, visibility=0.0, presence=0.0),
#             Landmark(x=-0.04064403474330902, y=0.0278518907725811, z=-0.023080304265022278, visibility=0.0, presence=0.0),
#             Landmark(x=-0.019516903907060623, y=-0.004718794487416744, z=0.022706955671310425, visibility=0.0, presence=0.0),
#             Landmark(x=-0.03854118660092354, y=-0.026829317212104797, z=0.013633454218506813, visibility=0.0, presence=0.0),
#             Landmark(x=-0.056818149983882904, y=-0.041743338108062744, z=0.00798335112631321, visibility=0.0, presence=0.0),
#             Landmark(x=-0.07528021931648254, y=-0.056438982486724854, z=-0.015836378559470177, visibility=0.0, presence=0.0),
#             Landmark(x=-0.003286648541688919, y=-0.0065175434574484825, z=0.004782100208103657, visibility=0.0, presence=0.0),
#             Landmark(x=-0.022455234080553055, y=-0.03259669616818428, z=-0.012064073234796524, visibility=0.0, presence=0.0),
#             Landmark(x=-0.03784722089767456, y=-0.04815170168876648, z=-0.028059333562850952, visibility=0.0, presence=0.0),
#             Landmark(x=-0.05421184003353119, y=-0.0688016340136528, z=-0.04320758953690529, visibility=0.0, presence=0.0),
#             Landmark(x=0.012875700369477272, y=0.0012778001837432384, z=-0.015323982574045658, visibility=0.0, presence=0.0),
#             Landmark(x=-0.01567920111119747, y=-0.0025686235167086124, z=-0.028398297727108, visibility=0.0, presence=0.0),
#             Landmark(x=-0.03174154460430145, y=0.012949916534125805, z=-0.0293277520686388, visibility=0.0, presence=0.0),
#             Landmark(x=-0.04413805902004242, y=0.0314234122633934, z=-0.028064286336302757, visibility=0.0, presence=0.0),
#             Landmark(x=0.025250781327486038, y=0.021189434453845024, z=-0.027470724657177925, visibility=0.0, presence=0.0),
#             Landmark(x=0.000794973224401474, y=0.023424040526151657, z=-0.034258510917425156, visibility=0.0, presence=0.0),
#             Landmark(x=-0.02167709916830063, y=0.03132347762584686, z=-0.034330420196056366, visibility=0.0, presence=0.0),
#             Landmark(x=-0.036197151988744736, y=0.04055459052324295, z=-0.03508662432432175, visibility=0.0, presence=0.0),
#         ]
#     ],
# )
# GestureRecognizerResult(
#     gestures=[[Category(index=-1, score=0.503886342048645, display_name="", category_name="None")]],
#     handedness=[[Category(index=1, score=0.9847562313079834, display_name="Left", category_name="Left")]],
#     hand_landmarks=[
#         [
#             NormalizedLandmark(x=0.9249206781387329, y=0.9814525246620178, z=5.056771783529257e-07, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.841513454914093, y=0.9698724150657654, z=-0.0028154454194009304, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7830360531806946, y=0.9438164234161377, z=-0.015507380478084087, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7455893754959106, y=0.9253418445587158, z=-0.0344543419778347, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7267335653305054, y=0.9160909056663513, z=-0.0533311702311039, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7805366516113281, y=0.8466737866401672, z=-0.010565939359366894, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.725021243095398, y=0.7958099246025085, z=-0.03320488706231117, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.6897746324539185, y=0.7650381922721863, z=-0.04899713769555092, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.6645621061325073, y=0.7369540333747864, z=-0.05966254323720932, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.8131321668624878, y=0.8382458686828613, z=-0.028543489053845406, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7657265663146973, y=0.773754358291626, z=-0.05648873746395111, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7309700846672058, y=0.7330806255340576, z=-0.07647628337144852, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7074856758117676, y=0.700488269329071, z=-0.08808416873216629, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.8434152603149414, y=0.8485411405563354, z=-0.04743294045329094, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7762919664382935, y=0.828584611415863, z=-0.08201218396425247, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7385871410369873, y=0.8691006302833557, z=-0.0932077094912529, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7220090627670288, y=0.9060431718826294, z=-0.09293486177921295, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.864338219165802, y=0.8774932026863098, z=-0.06598381698131561, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7993975281715393, y=0.8831247091293335, z=-0.09133321791887283, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7599987983703613, y=0.9083620309829712, z=-0.09558176249265671, visibility=0.0, presence=0.0),
#             NormalizedLandmark(x=0.7351073026657104, y=0.930469810962677, z=-0.09373602271080017, visibility=0.0, presence=0.0),
#         ]
#     ],
#     hand_world_landmarks=[
#         [
#             Landmark(x=0.05591510236263275, y=0.07214608043432236, z=0.04334268718957901, visibility=0.0, presence=0.0),
#             Landmark(x=0.01897272653877735, y=0.058044321835041046, z=0.044269759207963943, visibility=0.0, presence=0.0),
#             Landmark(x=-0.00727539137005806, y=0.046263113617897034, z=0.03549172729253769, visibility=0.0, presence=0.0),
#             Landmark(x=-0.03017817623913288, y=0.03852538764476776, z=0.004589414689689875, visibility=0.0, presence=0.0),
#             Landmark(x=-0.038070499897003174, y=0.02988898567855358, z=-0.02295597270131111, visibility=0.0, presence=0.0),
#             Landmark(x=-0.01794418692588806, y=-0.004345500376075506, z=0.02375921607017517, visibility=0.0, presence=0.0),
#             Landmark(x=-0.036574218422174454, y=-0.024518482387065887, z=0.013068614527583122, visibility=0.0, presence=0.0),
#             Landmark(x=-0.05558256059885025, y=-0.03732888400554657, z=0.007349823135882616, visibility=0.0, presence=0.0),
#             Landmark(x=-0.0758395865559578, y=-0.04747989773750305, z=-0.015643415972590446, visibility=0.0, presence=0.0),
#             Landmark(x=-0.0027976068668067455, y=-0.006892207078635693, z=0.003859673161059618, visibility=0.0, presence=0.0),
#             Landmark(x=-0.02393849939107895, y=-0.03201446309685707, z=-0.01075805351138115, visibility=0.0, presence=0.0),
#             Landmark(x=-0.04230187088251114, y=-0.04519101232290268, z=-0.02659992128610611, visibility=0.0, presence=0.0),
#             Landmark(x=-0.06189281865954399, y=-0.06466030329465866, z=-0.03811999410390854, visibility=0.0, presence=0.0),
#             Landmark(x=0.01107918843626976, y=0.001342968549579382, z=-0.015801524743437767, visibility=0.0, presence=0.0),
#             Landmark(x=-0.017970610409975052, y=-0.0023244814947247505, z=-0.026808978989720345, visibility=0.0, presence=0.0),
#             Landmark(x=-0.034089695662260056, y=0.014298112131655216, z=-0.027318332344293594, visibility=0.0, presence=0.0),
#             Landmark(x=-0.044740110635757446, y=0.031156087294220924, z=-0.023544423282146454, visibility=0.0, presence=0.0),
#             Landmark(x=0.02293401211500168, y=0.021869368851184845, z=-0.027783973142504692, visibility=0.0, presence=0.0),
#             Landmark(x=-0.001945185475051403, y=0.024037037044763565, z=-0.03267320618033409, visibility=0.0, presence=0.0),
#             Landmark(x=-0.025192957371473312, y=0.032640717923641205, z=-0.029644576832652092, visibility=0.0, presence=0.0),
#             Landmark(x=-0.0366290807723999, y=0.04359882324934006, z=-0.02569407969713211, visibility=0.0, presence=0.0),
#         ]
#     ],
# )
