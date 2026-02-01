from loguru import logger
import cv2
import numpy as np

from src.schema import Landmark, LandmarkedFrame, LandmarkedHand, LandmarkedHandIndex, Settings, DepthThresholdLimit


def draw_circle(
    image,
    landmark: Landmark,
    depth_info: DepthThresholdLimit,
    image_dimensions: tuple[int, int],
    power: float = 0.3,
):
    width, height = image_dimensions

    # Integer pixel locations
    px = max(min(int(landmark.x * (width - 1)), width - 1), 0)
    py = max(min(int(landmark.y * (height - 1)), height - 1), 0)

    if landmark.z is None or landmark.z >= depth_info.limit:
        alpha = 0.0
    elif landmark.z <= depth_info.threshold:
        alpha = 1.0
    else:
        # Normalize distance from threshold
        t = (landmark.z - depth_info.threshold) / (depth_info.limit - depth_info.threshold)
        # Exponential falloff for drastic effect
        alpha = 1.0 - (t**power)

    # Map alpha to color (from dark blue to bright blue, for example)
    bright_color = np.array(depth_info.near_color, dtype=np.uint8)
    dark_color = np.array(depth_info.far_color, dtype=np.uint8)

    # Interpolate
    color = (dark_color * (1 - alpha) + bright_color * alpha).astype(np.uint8)
    cv2.circle(image, (px, py), radius=10, color=color.tolist(), thickness=-1)


def draw_landmarks(
    landmarked_hand: LandmarkedFrame,
    settings: Settings,
    image_dimensions: tuple[int, int] = (480, 480),
) -> np.ndarray:
    annotated_image = np.zeros(
        (image_dimensions[1], image_dimensions[0], 3),
        dtype=np.uint8,
    )

    for hand in landmarked_hand.hands:
        for i, landmark in enumerate(hand.landmarks):
            if i in LandmarkedHandIndex.finger_tip_indices():
                draw_circle(
                    annotated_image,
                    landmark,
                    settings.recognition.click_depth_threshold,
                    image_dimensions,
                    1,
                )
                continue

            draw_circle(
                annotated_image,
                landmark,
                settings.recognition.move_depth_threshold,
                image_dimensions,
            )

    return annotated_image
