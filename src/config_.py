from dataclasses import dataclass
import cv2
from mediapipe.tasks.python import vision  # type: ignore[import]
import tkinter as tk
from tkinter import StringVar, BooleanVar, IntVar, DoubleVar


class LocationIdentifier:
    """Normalized locations for the camera view.
    - x, y start at top left corner of the screen.
    - width, height expand rightward and downward respectively.
    - All values should be between 0 and 1."""

    def __init__(self, master: tk.Tk, x: float, y: float, width: float, height: float) -> None:
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
            raise ValueError("All values must be between 0 and 1.")
        self.x = DoubleVar(master, value=x)
        self.y = DoubleVar(master, value=y)
        self.width = DoubleVar(master, value=width)
        self.height = DoubleVar(master, value=height)

    def start_coordinates(self, width: int, height: int) -> tuple[int, int]:
        return int(self.x.get() * width), int(self.y.get() * height)

    def end_coordinates(self, width: int, height: int) -> tuple[int, int]:
        return int((self.x.get() + self.width.get()) * width), int((self.y.get() + self.height.get()) * height)

    def coord_in_location(self, x: float, y: float) -> bool:
        """Check if the coordinates are within the location."""
        return self.x.get() <= x and x <= self.x.get() + self.width.get() and self.y.get() <= y and y <= self.y.get() + self.height.get()


class HGDCameraConfig:
    def __init__(self, master: tk.Tk) -> None:
        self.camera_base_vertical_resolution = 1080  # should not change
        self.camera_size = DoubleVar(master, value=0.1)
        self.aspect_ratio = DoubleVar(master, value=16 / 14)

    def camera_vertical_resolution(self):
        return int(self.camera_base_vertical_resolution * (1 - self.camera_size.get()))

    def camera_horizontal_resolution(self):
        return int(self.camera_vertical_resolution() * self.aspect_ratio.get())


class HGDMouseMovementConfig:
    def __init__(self, master: tk.Tk) -> None:
        self.mouse_base_sensitivity: int = 1000  # should not change

        self.mouse_deadzone: LocationIdentifier = LocationIdentifier(master, 0.35, 0.35, 0.3, 0.3)
        self.mouse_sensitivity = DoubleVar(master, value=7)  # scale 1-10
        self.repeated_frames_before_action = IntVar(master, value=2)  # scale 1-4
        self.repeated_frames_to_stop_action = IntVar(master, value=5)  # scale 1-10


class HGDScrollConfig:
    def __init__(self, master: tk.Tk) -> None:
        self.scroll_base_sensitivity: int = 20  # should not change

        self.scroll_zone: LocationIdentifier = LocationIdentifier(master, 0.5, 0.35, 0.15, 0.3)
        self.scroll_sensitivity = DoubleVar(master, value=5)  # scale 1-10


@dataclass
class HGDAnnotationConfig:
    """Configuration for the annotation text on the camera view."""

    margin: int = 10
    """Space between annotation text and hand"""
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_size: int = 1
    font_thickness: int = 1
    line_style: int = cv2.LINE_AA

    hand_text_colour: tuple = (88, 205, 54)
    box_colour: tuple = (0, 0, 255)


@dataclass
class Consts:
    model_asset_path: str = "./model/trained_mediapipe_gesture_recognizer.task"
    num_detected_hands: int = 1
    running_mode = vision.RunningMode.VIDEO


class Gestures:
    """Gesture-to-action mapping configuration."""

    def __init__(self, master) -> None:
        self.enable_palm_direction_checking_for_exit: BooleanVar = BooleanVar(master, value=True)
        self.show_current_gesture: BooleanVar = BooleanVar(master, value=False)
        self.exit_gesture: StringVar = StringVar(master, value="stop_inverted")
        self.left_click_gesture: StringVar = StringVar(master, value="ok")
        self.right_click_gesture: StringVar = StringVar(master, value="peace")
        self.drag_gesture: StringVar = StringVar(master, value="fist")
        self.scroll_up_gesture: StringVar = StringVar(master, value="two_up")
        self.scroll_down_gesture: StringVar = StringVar(master, value="two_up_inverted")

        self.available_gestures: list = [
            "fist",
            "ok",
            "one",
            "palm",
            "peace",
            "peace_inverted",
            "rock",
            "stop",
            "stop_inverted",
            "two_up",
            "two_up_inverted",
        ]


class HGDConfig:

    def __init__(self, master: tk.Tk) -> None:
        self.camera_config: HGDCameraConfig = HGDCameraConfig(master)
        self.mouse_movement_config: HGDMouseMovementConfig = HGDMouseMovementConfig(master)
        self.scroll_config: HGDScrollConfig = HGDScrollConfig(master)
        self.annotation_config: HGDAnnotationConfig = HGDAnnotationConfig()
        self.consts: Consts = Consts()
        self.gestures: Gestures = Gestures(master)


all_config_classes = HGDConfig | HGDMouseMovementConfig | HGDScrollConfig | HGDCameraConfig | Gestures
