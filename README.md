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

