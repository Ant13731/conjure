# Conjure

Welcome to Conjure, a hands-only system for controlling your computer! Requires a webcam.

## Features

- Mouse movement according to hand position
- Specific gestures to control common inputs (eg. "ok" for left click, "peace" for right click, "fist" for click-and-drag)
- Fully configurable input with GUI

### Known Bugs/Future Features

- Add voice activation for speech-to-text typing
- Rework mouse sensitivity to be better at extreme values (setting sensitivity too low stops all movement)
- Rework configuration setup, allow configuration options to be stored between executions
- Rework GUI
  - Combine configuration GUI and camera feed
  - Move away from Tkinter for a more polished look/feel

## Datasets

- CNN Landmark model using kaggle.com/datasets/soumikrakshit/rhp-dataset
- Classifier model over landmark model using a modified version of huggingface.co/cj-mills/hagrid-sample-30k-384p (only the `images` folder, excluding `call, dislike, four, like, mute, three, three2` subfolders)

To convert Google's Mediapipe hand landmark model into a gesture classifier, we use mediapipe_maker_tools. This library only functions on linux, and the dataset must be downloaded and prepared as described in the previous point (ie., moving the `call, dislike, four, like, mute, three, three2` image subfolders into `image/none`).

## Usage

Run

```bash
python main.py
```

To train your own classifier model on top of Google's Mediapipe gesture landmarker, see `model/mediapipe_classifier_training.py`

### Controls

Hand movement in the camera’s 2D projection of 3D space coincides with mouse movement across a computer’s monitor. For fine-grained movements and other interactions, Conjure creates a deadzone at the center of the screen, where all hand movements are ignored. Then, as a hand moves out of the deadzone toward the borders of the screen, the mouse will move with increasing velocity in the hand’s general direction.

The following is a table detailing the correspondence between gestures and Conjure's actions:

| Gesture                                      | Action           |
| -------------------------------------------- | ---------------- |
| ok                                           | Left click       |
| peace                                        | Right click      |
| clenching fist                               | click and hold   |
| backhanded stop                              | exit the program |
| two fingers up, palm facing towards camera   | scroll up        |
| two fingers up, palm facing away from camera | scroll down      |

Note that all of these controls are configurable through the GUI.

### References

See the [report](https://github.com/Ant13731/conjure/blob/main/docs/report.pdf) for a list of references used in this project.
