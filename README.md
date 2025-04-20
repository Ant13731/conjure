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

### Controls

- TODO <!--TODO----------------------------------------------------------------------------->

# References

- https://opencv.org/
- https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
- pyautogui
- https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer
- https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer
- https://arxiv.org/pdf/1705.01389
- https://arxiv.org/pdf/2206.08219
- https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p
- https://arxiv.org/html/2412.01508v1#bib.bib4 or https://arxiv.org/pdf/2412.01508
