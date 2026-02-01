from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, IntEnum, Enum
from typing import Any, Self, ClassVar


# MARK: Hand recognition schemas
## Enums
class EnumExtention(Enum):
    _ignore_ = ["_default"]
    _default: ClassVar[Self]

    @classmethod
    def from_(cls, value: str | int | Any) -> Self:
        try:
            return cls(value)
        except ValueError:
            return cls._default  # type: ignore


class Orientation(IntEnum, EnumExtention):
    unknown = 0
    portrait = 1
    portrait_upside_down = 2
    landscape_left = 3
    landscape_right = 4
    face_up = 5
    face_down = 6

    _default = unknown  # type: ignore


class Gesture(StrEnum, EnumExtention):
    unknown = "unknown"
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

    _default = unknown  # type: ignore


class Handedness(StrEnum, EnumExtention):
    unknown = "unknown"
    left = "left"
    right = "right"

    _default = unknown  # type: ignore

    def swap(self) -> Handedness:
        if self == Handedness.left:
            return Handedness.right
        elif self == Handedness.right:
            return Handedness.left
        else:
            return Handedness.unknown


class LandmarkedHandIndex(IntEnum, EnumExtention):
    wrist = 0
    thumb_CMC = 1
    thumb_MCP = 2
    thumb_IP = 3
    thumb_tip = 4
    index_MCP = 5
    index_PIP = 6
    index_DIP = 7
    index_tip = 8
    middle_MCP = 9
    middle_PIP = 10
    middle_DIP = 11
    middle_tip = 12
    ring_MCP = 13
    ring_PIP = 14
    ring_DIP = 15
    ring_tip = 16
    pinky_MCP = 17
    pinky_PIP = 18
    pinky_DIP = 19
    pinky_tip = 20

    _default = wrist  # type: ignore

    @classmethod
    def finger_tip_indices(cls) -> set[LandmarkedHandIndex]:
        return {
            LandmarkedHandIndex.thumb_tip,
            LandmarkedHandIndex.index_tip,
            LandmarkedHandIndex.middle_tip,
            LandmarkedHandIndex.ring_tip,
            LandmarkedHandIndex.pinky_tip,
        }

    @classmethod
    def connections(cls) -> list[tuple[int, int]]:
        return [
            # Thumb
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            # Index finger
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            # Middle finger
            (9, 10),
            (10, 11),
            (11, 12),
            (5, 9),
            # Ring finger
            (13, 14),
            (14, 15),
            (15, 16),
            (9, 13),
            # Pinky
            (17, 18),
            (18, 19),
            (19, 20),
            (13, 17),
            # Palm
            (0, 17),
        ]


## Landmark classes
@dataclass
class Landmark:
    x: float
    y: float
    z: float | None
    relative_depth: float | None
    visible: bool | None

    @classmethod
    def from_(cls, value: dict[str, Any]) -> Landmark:
        return cls(
            x=value["x"],
            y=value["y"],
            z=value.get("z"),
            relative_depth=value.get("relativeDepth"),
            visible=value.get("visible"),
        )


@dataclass
class LandmarkedHand:
    handedness: Handedness
    landmarks: list[Landmark]
    gesture: Gesture
    handedness_confidence: float
    gesture_confidence: float

    @classmethod
    def from_(cls, value: dict[str, Any]) -> LandmarkedHand:
        landmarks = list(map(Landmark.from_, value.get("landmarks", [])))
        return cls(
            handedness=Handedness.from_(value["handedness"]),
            landmarks=landmarks,
            gesture=Gesture.from_(value["gesture"]),
            handedness_confidence=value["handednessConfidence"],
            gesture_confidence=value["gestureConfidence"],
        )


@dataclass
class LandmarkedFrame:
    hands: list[LandmarkedHand]
    timestamp: int
    orientation: Orientation

    @classmethod
    def from_(cls, value: dict[str, Any]) -> LandmarkedFrame:
        hands = list(map(LandmarkedHand.from_, value.get("hands", [])))

        return cls(
            hands=hands,
            timestamp=value["timestamp"],
            orientation=Orientation.from_(value["orientation"]),
        )


# MARK: Trackpad schemas
# MARK: Settings schemas
class ConnectionMode(StrEnum, EnumExtention):
    web_rtc = "web_rtc"
    tcp = "tcp"
    udp = "udp"

    _default = web_rtc  # type: ignore


class OperationMode(StrEnum, EnumExtention):
    trackpad = "trackpad"
    hand_recognition = "hand_recognition"
    hand_recognition_demo = "hand_recognition_demo"

    _default = hand_recognition  # type: ignore


@dataclass
class GeneralSettings:
    connection_mode: ConnectionMode
    operation_mode: OperationMode
    host_friendly_name: str


@dataclass
class TrackpadSettings:
    sensitivity: float
    invert_x: bool
    invert_y: bool


@dataclass
class DepthThresholdLimit:
    threshold: float
    limit: float
    near_color: list[int]
    far_color: list[int]


@dataclass
class RecognitionSettings:
    num_hands: int
    landmark_depth_pixel_radius: int
    min_depth: float
    max_depth: float

    click_depth_threshold: DepthThresholdLimit
    move_depth_threshold: DepthThresholdLimit


@dataclass
class Settings:
    general: GeneralSettings
    trackpad: TrackpadSettings
    recognition: RecognitionSettings

    @classmethod
    def from_(cls, value: dict[str, dict]) -> Settings:
        general_settings = value.get("generalSettings", {})
        host_settings = value.get("hostListSettings", {}).get("currentHost", {})
        trackpad_settings = value.get("trackpadSettings", {})
        recognition_settings = value.get("recognitionSettings", {})

        connection_mode = general_settings["connectionMode"]
        operation_mode = general_settings["operationMode"]

        host_friendly_name = host_settings.get("friendlyName")

        sensitivity = trackpad_settings["sensitivity"]
        invert_x = trackpad_settings["invertX"]
        invert_y = trackpad_settings["invertY"]

        num_hands = recognition_settings["numHands"]
        landmark_depth_pixel_radius = recognition_settings["landmarkDepthPixelRadius"]
        min_depth = recognition_settings["minDepth"]
        max_depth = recognition_settings["maxDepth"]

        click_depth_threshold = recognition_settings["clickDepthThreshold"]
        click_depth_limit = recognition_settings["clickDepthLimit"]
        click_depth_near_color = list(recognition_settings.get("fingerTipColorNear", {"r": 255, "g": 200, "b": 0}).values())
        click_depth_far_color = list(recognition_settings.get("fingerTipColorFar", {"r": 50, "g": 40, "b": 0}).values())

        move_depth_threshold = recognition_settings["moveDepthThreshold"]
        move_depth_limit = recognition_settings["moveDepthLimit"]
        move_depth_near_color = list(recognition_settings.get("jointColorNear", {"r": 255, "g": 255, "b": 255}).values())
        move_depth_far_color = list(recognition_settings.get("jointColorFar", {"r": 0, "g": 0, "b": 0}).values())

        return cls(
            general=GeneralSettings(
                connection_mode=ConnectionMode.from_(connection_mode),
                operation_mode=OperationMode.from_(operation_mode),
                host_friendly_name=host_friendly_name,
            ),
            trackpad=TrackpadSettings(
                sensitivity=sensitivity,
                invert_x=invert_x,
                invert_y=invert_y,
            ),
            recognition=RecognitionSettings(
                num_hands=num_hands,
                landmark_depth_pixel_radius=landmark_depth_pixel_radius,
                min_depth=min_depth,
                max_depth=max_depth,
                click_depth_threshold=DepthThresholdLimit(
                    threshold=click_depth_threshold,
                    limit=click_depth_limit,
                    near_color=click_depth_near_color,
                    far_color=click_depth_far_color,
                ),
                move_depth_threshold=DepthThresholdLimit(
                    threshold=move_depth_threshold,
                    limit=move_depth_limit,
                    near_color=move_depth_near_color,
                    far_color=move_depth_far_color,
                ),
            ),
        )

    # TODO: add methods to update settings
