TODO Features:
- ios-client
    - MainView:
        - trackpad should take up bottom 3/4 - 5/6 of the screen
- server
    - trackpad driver


Bugs:
- gesture recognition only works in camera-to-the-right landscape mode
    - so the ground position of the camera frame matters to mediapipe
    - to fix, we could rotate the image 90 degrees to the right when vertical, and 180 when left landscape
- movement is very choppy, which we didnt see when streaming through tcp/udp...
    - skeleton overlay looks smooth though
    - could try sending data through TCP/UDP but still do processing on-device
