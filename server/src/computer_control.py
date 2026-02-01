from dataclasses import dataclass, field
from threading import Thread, Event
from queue import Queue, Empty

from loguru import logger
import pyautogui as pg

from src.schema import Gesture, Landmark, LandmarkedFrame, LandmarkedHandIndex, Settings


class LimitedLengthList[T](list[T]):
    def __init__(self, max_length: int):
        super().__init__()
        self.max_length = max_length

    def append(self, item):
        if len(self) >= self.max_length:
            self.pop(0)
        super().append(item)


@dataclass
class ComputerControl:
    end_event: Event

    thread: Thread | None = None
    queue: Queue[LandmarkedFrame] = field(default_factory=lambda: Queue(maxsize=1))

    settings: Settings | None = None

    cursor_velocity: tuple[float, float] = (0.0, 0.0)
    prev_cursor_position: Landmark | None = None

    is_dragging: bool = False
    last_n_left_click: LimitedLengthList[bool] = field(default_factory=lambda: LimitedLengthList(max_length=10))
    last_n_right_click: LimitedLengthList[bool] = field(default_factory=lambda: LimitedLengthList(max_length=10))
    last_n_gestures: LimitedLengthList[Gesture] = field(default_factory=lambda: LimitedLengthList(max_length=5))

    def is_left_click_on_cooldown(self) -> bool:
        return any(self.last_n_left_click)

    def is_right_click_on_cooldown(self) -> bool:
        return any(self.last_n_right_click)

    def sufficiently_repeated_before_action(self, gesture: Gesture, required_count: int = 2) -> bool:
        count = sum(1 for g in self.last_n_gestures if g == gesture)
        return count >= required_count

    def move_cursor_with_velocity(self, stop_when_below: float = 0.1) -> None:
        if abs(self.cursor_velocity[0]) < stop_when_below and abs(self.cursor_velocity[1]) < stop_when_below:
            return
        pg.moveRel(-self.cursor_velocity[0], -self.cursor_velocity[1], duration=0.1)

    def decay_cursor_velocity(self, decay_factor: float = 0.8) -> None:
        vx, vy = self.cursor_velocity
        self.cursor_velocity = (vx * decay_factor, vy * decay_factor)

    def increment_cursor_velocity(self, landmark: Landmark, scaling: float = 1000) -> None:
        if self.prev_cursor_position is None:
            logger.info("Cannot find previous cursor position, not incrementing cursor velocity")
            return

        dx = self.prev_cursor_position.x - landmark.x
        dy = self.prev_cursor_position.y - landmark.y
        dx *= scaling
        dy *= scaling

        self.cursor_velocity = (
            self.cursor_velocity[0] + dx,
            self.cursor_velocity[1] + dy,
        )

    def move_cursor(self, landmark: Landmark, scaling: float = 1500) -> None:
        if self.prev_cursor_position is None:
            logger.info("Cannot find previous cursor position, not moving cursor")
            return

        dx = self.prev_cursor_position.x - landmark.x
        dy = self.prev_cursor_position.y - landmark.y
        dx *= scaling
        dy *= scaling

        pg.moveRel(-dx, -dy, duration=0.1)

    def is_within_click_threshold(self, depth: float | None) -> bool:
        if depth is None or self.settings is None:
            return False
        return depth < self.settings.recognition.click_depth_threshold.threshold

    def is_within_move_threshold(self, depth: float | None) -> bool:
        if depth is None or self.settings is None:
            return False
        return depth < self.settings.recognition.move_depth_threshold.threshold

    def run(self) -> None:
        pg.PAUSE = 0
        logger.info("Starting computer control loop")
        while True:
            if self.end_event.is_set():
                logger.info("End event set, stopping computer control thread")
                break

            try:
                frame = self.queue.get_nowait()
            except Empty:
                continue

            logger.info(f"Got frame: {frame}")

            self.decay_cursor_velocity()
            self.move_cursor_with_velocity()

            if frame.hands == []:
                self.prev_cursor_position = None
                self.are_dragging = False
                pg.mouseUp(button="left")
                continue

            landmarked_hand = frame.hands[0]
            logger.info(f"Using landmarked hand: {landmarked_hand}")

            self.last_n_gestures.append(landmarked_hand.gesture)
            self.last_n_left_click.append(False)
            self.last_n_right_click.append(False)

            # When we see a palm, cancel all actions
            if landmarked_hand.gesture in (Gesture.palm, Gesture.stop, Gesture.stop_inverted) and self.sufficiently_repeated_before_action(landmarked_hand.gesture):
                logger.info("Palm detected, cancelling actions")
                self.cursor_velocity = (0, 0)
                self.are_dragging = False
                pg.mouseUp(button="left")
                self.prev_index_location = landmarked_hand.landmarks[LandmarkedHandIndex.index_tip]
                continue

            logger.info("Checkpoint 1")
            # Left Click
            if (
                landmarked_hand.gesture == Gesture.one
                and self.is_within_click_threshold(landmarked_hand.landmarks[LandmarkedHandIndex.index_tip].z)
                and self.sufficiently_repeated_before_action(landmarked_hand.gesture)
                and not self.is_left_click_on_cooldown()
            ):
                logger.info("Left click")
                pg.click(button="left")
                self.last_n_left_click.append(True)

            logger.info("Checkpoint 2")
            # Right Click
            if (
                landmarked_hand.gesture == Gesture.peace
                and self.is_within_click_threshold(landmarked_hand.landmarks[LandmarkedHandIndex.index_tip].z)
                and self.sufficiently_repeated_before_action(landmarked_hand.gesture)
                and not self.is_left_click_on_cooldown()
            ):
                logger.info("Right click")
                pg.click(button="right")
                self.last_n_right_click.append(True)

            logger.info("Checkpoint 3")
            # Click and hold for dragging
            if (
                landmarked_hand.gesture in (Gesture.ok, Gesture.fist)
                and self.is_within_click_threshold(landmarked_hand.landmarks[LandmarkedHandIndex.index_tip].z)
                and self.sufficiently_repeated_before_action(landmarked_hand.gesture)
                and not self.is_dragging
            ):
                logger.info("Starting left click drag")
                pg.mouseDown(button="left")
                self.is_dragging = True

            logger.info("Checkpoint 4")
            if not self.is_within_click_threshold(landmarked_hand.landmarks[LandmarkedHandIndex.index_tip].z) and self.is_dragging:
                logger.info("Ending left click drag")
                pg.mouseUp(button="left")
                self.is_dragging = False

            logger.info("Checkpoint 5")
            # Small, absolute movements
            if landmarked_hand.gesture == Gesture.one and self.is_within_move_threshold(landmarked_hand.landmarks[LandmarkedHandIndex.index_tip].z):
                logger.info("Moving cursor with small movements")
                self.move_cursor(landmarked_hand.landmarks[LandmarkedHandIndex.index_tip])

            logger.info("Checkpoint 6")
            # Sweeping, general movements
            if (
                landmarked_hand.gesture in (Gesture.two_up, Gesture.two_up_inverted)
                and self.is_within_move_threshold(landmarked_hand.landmarks[LandmarkedHandIndex.index_tip].z)
                and self.prev_cursor_position is not None
            ):
                logger.info("Incrementing cursor velocity")
                self.increment_cursor_velocity(landmarked_hand.landmarks[LandmarkedHandIndex.index_tip])

            logger.info("Checkpoint 7")
            self.prev_cursor_position = landmarked_hand.landmarks[LandmarkedHandIndex.index_tip]

    def start(self) -> None:
        self.thread = Thread(target=self.run, args=())
        self.thread.start()

    def update_settings(self, settings: Settings) -> None:
        logger.info("Receiving updated settings for computer control")
        self.settings = settings

    def receive_frame(self, frame: LandmarkedFrame) -> None:
        # try:
        #     self.queue.get_nowait()
        # except Empty:
        #     pass

        logger.info("Receiving frame for computer control")
        self.queue.put_nowait(frame)
