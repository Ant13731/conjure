import threading
import tkinter as tk

from src import gui
from src import computer_control
from src.config_ import HGDConfig

# Global variable is a hack but I need it to pass the camera control status between the start/stop buttons
camera_thread: None | threading.Thread = None
end_event: None | threading.Event = None


def start_camera(config: HGDConfig) -> None:
    """Start the camera thread."""
    global camera_thread
    global end_event
    if camera_thread is not None and camera_thread.is_alive():
        print("Camera thread is already running.")
        return
    print("Starting camera thread...")
    end_event = threading.Event()
    camera_thread = threading.Thread(target=computer_control.main_camera, args=(config, end_event))
    print("Starting camera thread...")
    camera_thread.start()


def stop_camera() -> None:
    """Stop the camera thread."""
    global camera_thread
    global end_event
    if camera_thread is None or not camera_thread.is_alive() or end_event is None:
        print("Camera thread is not running.")
        return

    end_event.set()
    print("Waiting for camera thread to finish...")
    camera_thread.join(1)
    if camera_thread.is_alive():
        # for some reason the thread only terminates after this function has finished running
        print("Camera thread may not have terminated properly.")
    else:
        print("Camera thread finished.")


def main():
    """Main function to start the GUI and camera."""
    print("Starting configuration GUI...")
    root, canvas, config = gui.gui()

    end_event = threading.Event()

    camera_button_frame = tk.LabelFrame(canvas, text="Camera Control", bg="white")
    camera_button_frame.pack(padx=2, pady=2, fill=tk.X)

    camera_start_button = tk.Button(camera_button_frame, text="Start Camera", command=lambda: start_camera(config))
    camera_start_button.pack(pady=10, padx=10, fill=tk.X)

    camera_stop_button = tk.Button(camera_button_frame, text="Stop Camera", command=lambda: stop_camera())
    camera_stop_button.pack(pady=10, padx=10, fill=tk.X)

    root.mainloop()

    if camera_thread is not None and camera_thread.is_alive():
        end_event.set()
        print("Waiting for camera thread to finish...")
        camera_thread.join()


if __name__ == "__main__":
    main()
