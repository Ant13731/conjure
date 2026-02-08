# Conjure

Welcome to Conjure, a controller-free system for controlling your computer! Requires an iPhone with FaceID.

## Features
<p align="center">
  <img src="docs/images/hand screen.png" alt="Conjure in action" width="200"/>
  <img src="docs/images/disconnected screen.png" alt="Home screen" width="200"/>
</p>

- Mouse movement according to hand position, controlled using the depth and RGB sensors in your front-facing phone camera
- Specific gestures to control common inputs ("index finger" for left click, "peace" for right click, "pinch" to click and drag)

## Usage

1. Upload the ios-client to your iPhone by building through XCode
2. Set up a [Tailscale net](https://tailscale.com/) on both your phone and computer
3. Run the python server file on the computer you wish to control

### Controls

Hand movement in the camera’s 2D projection of 3D space coincides with mouse movement across a computer’s monitor. Depth values relative to the camera's plane limit available actions. More details are available in the `docs`.

The following is a table detailing the correspondence between gestures and Conjure's actions:

| Area                | Gesture                                        | Action                                                      |
| ------------------- | ---------------------------------------------- | ----------------------------------------------------------- |
| Anywhere            | Palm / Stop                                    | Stop all movement                                           |
| Movement Plane      | One (pointing with index finger)               | Cursor tracks index finger movement precisely               |
|                     | Two-up (pointing with index and middle finger) | Cursor follows index finger movement with decaying velocity |
| Click-through Plane | One                                            | Mouse left-click                                            |
|                     | Peace                                          | Mouse right-click                                           |
|                     | OK (pinch)                                     | Mouse left-click and hold (for clicking+dragging)           |

### Architecture
High level architecture (certain modes may work wirelessly):
<p align="center">
  <img src="docs/images/highLevelArch.png" alt="Conjure in action" width="500"/>
</p>
Mental model of the system:
<p align="center">
  <img src="docs/images/mentalModel.png" alt="Conjure in action" width="300"/>
</p>
<p align="center">
  <img src="docs/images/index_back.jpg" alt="Conjure in action" height="130"/>
  <img src="docs/images/index_middle.jpg" alt="Conjure in action" height="130"/>
  <img src="docs/images/index_front.jpg" alt="Conjure in action" height="130"/>
</p>