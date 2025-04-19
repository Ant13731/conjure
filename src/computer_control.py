import cv2
import pyautogui as pg
import threading

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark  # type: ignore[import]
from mediapipe.tasks.python.components.containers.category import Category  # type: ignore[import]

from src.config_ import HGDConfig
from src import gesture_classifier_model


def start_camera() -> cv2.VideoCapture:
    """Handle live camera feed.

    Reference:
    - https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
    """
    cv2.namedWindow("Webcam Preview")
    vc = cv2.VideoCapture(0)

    if not vc.isOpened():
        raise Exception("Error: Could not open video device.")

    return vc


def configure_camera(vc: cv2.VideoCapture, config: HGDConfig) -> None:
    """Resize camera"""
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_config.camera_horizontal_resolution())
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_config.camera_vertical_resolution())


def is_palm_facing_camera(hand_landmarks: list[NormalizedLandmark], is_left_hand: bool) -> bool:
    """Check if the inside of a detected palm is facing towards the camera.

    This is done by checking the relative positions of the thumb, palm, and pinky (tested and recorded manually)."""

    palm_facing_camera = True

    thumb_left_of_pinky = hand_landmarks[3].x < hand_landmarks[17].x
    thumb_above_pinky = hand_landmarks[3].y < hand_landmarks[17].y

    thumb_left_of_palm = hand_landmarks[3].x < hand_landmarks[0].x
    thumb_above_palm = hand_landmarks[3].y < hand_landmarks[0].y

    palm_left_of_pinky = hand_landmarks[0].x < hand_landmarks[17].x
    palm_above_pinky = hand_landmarks[0].y < hand_landmarks[17].y

    record = (
        thumb_left_of_pinky,
        thumb_above_pinky,
        thumb_left_of_palm,
        thumb_above_palm,
        palm_left_of_pinky,
        palm_above_pinky,
    )

    # Tested the position results manually, copied them here
    # This could be done through a series of if statements or boolean logic,
    # but this approach is faster and empirically guaranteed to work for open palms
    palm_facing_camera_positions = [
        (False, False, True, False, False, False),
        (False, False, True, False, False, True),
        (False, True, False, False, True, True),
        (False, True, False, True, True, True),
        (True, False, True, False, False, False),
        (True, False, True, False, True, False),
        (True, False, True, True, False, False),
        (True, False, True, True, True, False),
        (True, True, False, True, True, False),
        (True, True, False, True, True, True),
        (True, True, True, True, True, False),
    ]

    palm_facing_camera = record in palm_facing_camera_positions
    if not is_left_hand:
        palm_facing_camera = not palm_facing_camera

    return palm_facing_camera


def main_camera(config: HGDConfig, end_event: threading.Event) -> None:
    """Main function for the camera thread."""
    vc = start_camera()
    configure_camera(vc, config)
    detector = gesture_classifier_model.get_mediapipe_model(config)
    pg.PAUSE = 0

    last_10_gestures = []
    last_10_left_click = []
    last_10_right_click = []
    last_10_left_down = []

    while True:
        ret, frame = vc.read()
        if not ret:
            raise Exception(f"Error: Could not read frame. Got {frame}")

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect (moving hand to the right will move the cursor to the right)

        detection_result, annotated_image, hand_detected = gesture_classifier_model.predict(frame, detector, config)

        cv2.imshow("Webcam Preview", annotated_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit key entered. Exiting...")
            break

        if end_event.is_set():
            print("End event set. Exiting...")
            break

        if not hand_detected:
            continue

        gesture: Category = detection_result.gestures[0][0]
        hand_landmarks: list[NormalizedLandmark] = detection_result.hand_landmarks[0]
        is_left_hand: bool = detection_result.handedness[0][0].category_name == "Left"

        # Keep a record of the last 10 iterations
        # (used to increase reliability of clicking, scrolling, etc. since input is a continuous stream)
        last_10_gestures.append(gesture.category_name)
        if len(last_10_gestures) > 10:
            last_10_gestures = last_10_gestures[-10:]

        last_10_left_click.append(False)
        if len(last_10_left_click) > 10:
            last_10_left_click = last_10_left_click[-10:]

        last_10_right_click.append(False)
        if len(last_10_right_click) > 10:
            last_10_right_click = last_10_right_click[-10:]

        last_10_left_down.append(False)
        if len(last_10_left_down) > 10:
            last_10_left_down = last_10_left_down[-10:]

        # Used for a bunch of gestures later on
        x_index_tip = hand_landmarks[8].x
        y_index_tip = hand_landmarks[8].y
        x_index_tip_centered = x_index_tip - 0.5
        y_index_tip_centered = y_index_tip - 0.5

        palm_facing_camera = False
        if config.gestures.enable_palm_direction_checking_for_exit.get():
            palm_facing_camera = is_palm_facing_camera(hand_landmarks, is_left_hand)

        print(gesture)

        # Stop the camera detection if we see the stop_inverted gesture
        if (
            gesture.category_name == config.gestures.exit_gesture.get()
            and not palm_facing_camera
            and last_10_gestures[5:].count(config.gestures.exit_gesture.get()) > config.mouse_movement_config.repeated_frames_before_action.get()
        ):
            print("Backwards open palm detected, exiting program...")
            break

        # Add pinch gesture to left click
        # only click once every 10 frames, reduces chances of double click or mis-click
        # make sure you see 2 frames of clicking (basically) in a row
        if (
            gesture.category_name == config.gestures.left_click_gesture.get()
            and last_10_gestures[5:].count(config.gestures.left_click_gesture.get()) > config.mouse_movement_config.repeated_frames_before_action.get()
            and not any(last_10_left_click)
        ):
            pg.click(button="left")
            last_10_left_click.append(True)

        # Add open palm to right click
        if (
            gesture.category_name == config.gestures.right_click_gesture.get()
            and last_10_gestures[5:].count(config.gestures.right_click_gesture.get()) > config.mouse_movement_config.repeated_frames_before_action.get()
            and not any(last_10_right_click)
        ):
            pg.click(button="right")
            last_10_right_click.append(True)

        # Add closed fist to click and drag
        if gesture.category_name == config.gestures.drag_gesture.get():
            # If already in drag motion, keep button held down
            if any(last_10_left_down):
                last_10_left_down.append(True)
            # Start drag motion
            if last_10_gestures.count(config.gestures.drag_gesture.get()) > config.mouse_movement_config.repeated_frames_to_stop_action.get() and not any(last_10_left_down):
                pg.mouseDown(button="left")
                last_10_left_down.append(True)
        # Stop drag motion
        if last_10_gestures.count(config.gestures.drag_gesture.get()) < config.mouse_movement_config.repeated_frames_to_stop_action.get():
            pg.mouseUp(button="left")
            last_10_left_down = [False] * 10  # kind of a hack but it works

        # Add two fingers up to scroll (in all directions)
        # if x_index_tip_centered > 0 and x_index_tip_centered < SCROLL_ZONE and y_index_tip_centered > -SCROLL_ZONE and y_index_tip_centered < SCROLL_ZONE:
        if config.scroll_config.scroll_zone.coord_in_location(x_index_tip, y_index_tip):
            if gesture.category_name in [config.gestures.scroll_up_gesture.get(), config.gestures.scroll_down_gesture.get()]:
                height_between_index_and_palm = max(abs(hand_landmarks[0].y - hand_landmarks[8].y), 0.2)
                if gesture.category_name == config.gestures.scroll_down_gesture.get():
                    height_between_index_and_palm = -height_between_index_and_palm
                # effectively makes scrolling faster the closer you are
                pg.scroll(int(height_between_index_and_palm * config.scroll_config.scroll_sensitivity.get() * config.scroll_config.scroll_base_sensitivity))

        # if x_centered and y_centered are in the deadzone, do nothing
        if not config.mouse_movement_config.mouse_deadzone.coord_in_location(x_index_tip, y_index_tip):
            # Move hands like a cursor
            x_neg = -1 if x_index_tip_centered < 0 else 1
            y_neg = -1 if y_index_tip_centered < 0 else 1

            x_moverel = x_neg * (abs(x_index_tip_centered) ** (10 - config.mouse_movement_config.mouse_sensitivity.get())) * config.mouse_movement_config.mouse_base_sensitivity
            y_moverel = y_neg * (abs(y_index_tip_centered) ** (10 - config.mouse_movement_config.mouse_sensitivity.get())) * config.mouse_movement_config.mouse_base_sensitivity
            pg.moveRel(x_moverel, y_moverel, duration=0.1)

    pg.mouseUp(button="left")
    vc.release()
    cv2.destroyAllWindows()
