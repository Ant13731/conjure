from dataclasses import dataclass, field
from threading import Thread, Event
from queue import Queue, Empty

from src.schema import LandmarkedFrame, Settings


@dataclass
class ComputerControl:
    stop_event: Event

    thread: Thread | None = None
    queue: Queue[LandmarkedFrame] = field(default_factory=lambda: Queue(maxsize=1))

    settings: Settings | None = None

    def start(self) -> None:
        self.thread = Thread(target=self.run, args=())
        self.thread.start()

    def run(self) -> None: ...

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    def recieve_frame(self, frame: LandmarkedFrame) -> None:
        try:
            self.queue.get_nowait()
        except Empty:
            pass

        self.queue.put_nowait(frame)
