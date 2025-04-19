import tkinter as tk
from functools import singledispatch

import config_


def field_to_label(field: str) -> str:
    """Convert a field name to a label for the GUI."""
    return field.replace("_", " ").title()


def slider(
    frame: tk.Frame | tk.LabelFrame,
    config: config_.all_config_classes,
    from_: int,
    to: int,
    resolution: float,
    field: str,
) -> tk.Scale:
    """Create a floating point slider.

    Args:
    - frame: The frame to place the slider in.
    - config: The configuration object to update.
    - from_: The minimum value of the slider.
    - to: The maximum value of the slider.
    - resolution: The resolution of the slider.
    - field: The field of the config object to update (used also for display names)."""
    scale = tk.Scale(
        frame,
        from_=from_,
        to=to,
        orient=tk.HORIZONTAL,
        resolution=resolution,
        label=field_to_label(field),
        variable=getattr(config, field),
    )
    scale.pack(pady=5, padx=5, fill=tk.X)
    # scale.set(getattr(config, field))
    # scale.bind("<Motion>", lambda event: setattr(config, field, scale.get()))
    return scale


def sliders_location(
    frame: tk.Frame | tk.LabelFrame,
    location: config_.LocationIdentifier,
    parent_field: str,
) -> None:
    """Creates 4 sliders according to a LocationIdentifier's x, y, width, and height attributes.

    Args:
    - frame: The frame to place the sliders in.
    - location: The LocationIdentifier object to update.
    - parent_field: The field name of the parent object (used for display names)."""

    for attr in ["x", "y", "width", "height"]:
        scale = tk.Scale(
            frame,
            from_=0,
            to=1,
            orient=tk.HORIZONTAL,
            resolution=0.01,
            label=field_to_label(parent_field + "_" + attr),
            variable=getattr(location, attr),
        )
        scale.pack(pady=5, padx=5, fill=tk.X)
        # scale.set(getattr(location, attr))
        # scale.bind("<Motion>", lambda event: setattr(location, attr, scale.get()))


@singledispatch
def generate_config_widget(config: config_.all_config_classes, config_frame: tk.Frame) -> None:
    """Generate widgets for each group in the configuration tree."""
    raise NotImplementedError(f"Unsupported config type: {type(config)}")


@generate_config_widget.register(config_.HGDConfig)
def _(config: config_.HGDConfig, config_frame: tk.Frame) -> None:
    generate_config_widget(config.camera_config, config_frame)
    generate_config_widget(config.mouse_movement_config, config_frame)
    generate_config_widget(config.scroll_config, config_frame)
    generate_config_widget(config.gestures, config_frame)
    # generate_config_widget(config.annotation_config, config_frame)
    # generate_config_widget(config.consts, config_frame)


@generate_config_widget.register(config_.HGDCameraConfig)
def _(config: config_.HGDCameraConfig, config_frame: tk.Frame) -> None:
    camera_frame = tk.LabelFrame(config_frame, text="Camera Configuration", bg="white")
    camera_frame.pack(pady=10, padx=10, fill=tk.X)

    text_box = tk.Label(camera_frame, text="Restart camera for changes to take effect")
    text_box.pack(pady=5, padx=5, fill=tk.X)

    for field in ["camera_size", "aspect_ratio"]:
        slider(
            camera_frame,
            config,
            from_=0,
            to=4 if field == "aspect_ratio" else 1,
            resolution=0.01,
            field=field,
        )


@generate_config_widget.register(config_.HGDMouseMovementConfig)
def _(config: config_.HGDMouseMovementConfig, config_frame: tk.Frame) -> None:
    mouse_frame = tk.LabelFrame(config_frame, text="Mouse Movement Configuration", bg="white")
    mouse_frame.pack(pady=10, padx=10, fill=tk.X)

    slider(
        mouse_frame,
        config,
        from_=1,
        to=10,
        resolution=0.1,
        field="mouse_sensitivity",
    )
    sliders_location(
        mouse_frame,
        config.mouse_deadzone,
        "mouse_deadzone",
    )

    for field in ["repeated_frames_before_action", "repeated_frames_to_stop_action"]:
        slider(
            mouse_frame,
            config,
            from_=1,
            to=9 if field == "repeated_frames_to_stop_action" else 4,
            resolution=1,
            field=field,
        )


@generate_config_widget.register(config_.HGDScrollConfig)
def _(config: config_.HGDScrollConfig, config_frame: tk.Frame) -> None:
    scroll_frame = tk.LabelFrame(config_frame, text="Scroll Configuration", bg="white")
    scroll_frame.pack(pady=10, padx=10, fill=tk.X)

    slider(
        scroll_frame,
        config,
        from_=1,
        to=10,
        resolution=0.1,
        field="scroll_sensitivity",
    )
    sliders_location(
        scroll_frame,
        config.scroll_zone,
        "scroll_deadzone",
    )


@generate_config_widget.register(config_.Gestures)
def _(config: config_.Gestures, config_frame: tk.Frame) -> None:
    gestures_frame = tk.LabelFrame(config_frame, text="Gestures", bg="white")
    gestures_frame.pack(pady=10, padx=10, fill=tk.X)

    for gesture in [
        "exit_gesture",
        "scroll_up_gesture",
        "scroll_down_gesture",
        "left_click_gesture",
        "right_click_gesture",
        "drag_gesture",
    ]:
        label_frame = tk.LabelFrame(gestures_frame, text=field_to_label(gesture), bg="white")
        label_frame.pack(pady=5, padx=5, fill=tk.X)
        gesture_frame = tk.OptionMenu(label_frame, getattr(config, gesture), *config.available_gestures)
        gesture_frame.pack(pady=5, padx=5, fill=tk.X)

    checkbox = tk.Checkbutton(
        gestures_frame,
        text="Enable palm direction checking for exit",
        variable=config.enable_palm_direction_checking_for_exit,
        onvalue=True,
        offvalue=False,
    )
    checkbox.pack(pady=5, padx=5, fill=tk.X)


def gui() -> tuple[tk.Tk, tk.Canvas, config_.HGDConfig]:
    """Create the tkinter GUI with default configurations but do not run it.

    The GUI will update the returned configuration object - other threads may read from this object to get the current configuration."""

    # Make a simple configuration window to accompany the video output
    root = tk.Tk()
    root.title("Hand Gesture Recognition")
    root.geometry("400x600")
    root.configure(bg="white")

    canvas = tk.Canvas(root, bg="white")
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(root, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.config(yscrollcommand=scrollbar.set)

    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    canvas.bind("<Configure>", on_configure)
    canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

    config_frame = tk.Frame(canvas, bg="white")
    canvas.create_window((0, 0), window=config_frame, anchor="nw", width=380)

    padding = tk.Frame(config_frame, bg="white")
    padding.pack(pady=80, fill=tk.X)

    config = config_.HGDConfig(root)
    generate_config_widget(config, config_frame)

    return root, canvas, config
