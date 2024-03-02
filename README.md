# workout-counter
Workout counter is an application that takes camera input while you're doing an exercise and counts repetitions for you out loud as you do them.

<img src="./img/body_part_detection.png" width=200>

Here's how bicep curls look as a function of the coordinates X and Y of the subject's left wrist in time as he's doing the exercise:

![bicep curls](./img/bicep_curls.png)

## State of the work: Abandoned :(
The application is in an MVP state. Here's how it works:
    1. The app reads camera input from the device you're using.
    2. The camera feed is piped into [MoveNet](https://www.tensorflow.org/hub/tutorials/movenet) to detect the body position 60 times a second.
    3. A predefined limb is detected and its X and Y position are tracked, relative to the top left corner of the camera feed. We assume the camera is stationary.
    4. The descrete values of the position X in time, and the position Y in time are smoothened into functions.
    5. When peaks and valleys with similar amplitudes are detected twice in a row, they are treated as exercise repetitions.

## How did the app come to be?
A friend told me she has a problem in the gym; when she's focusing on the technique and breathing while doing an exercise, she has trouble keep track of the number of repetitions she did. I thought that was an interesting problem, and wanted to solve it with computer vision. We formed a 6 person team (2 software engineers, 2 computer vision students, a neuroscientist, and a personal trainer) to do a 48 hour hackaton over the weekend. After a bit of research and whiteboarding we came up with an MVP you can try for yourself, albeit a bit finicky.

## Here are some other diagrams of X and Y positions of the relevant joint
![pullups](./img/pullups.png)
![situps](./img/situps.png)
![tricep extensions](./img/tricep_extensions.png)

![bicep curls count](./img/bicep_curls_peaks_and_valleys.png)

## How to run
1. install python
2. install pip
3. install requirements `pip install -r requirements.txt`
4. (optional) find some pre-recorded videos
5. edit `app.py` (bottom of the file) to choose input (first argument of main) -- `0` means camera input, or list a path to the video file you want to analyse
6. change the joint that is tracked in `app.py` with the `body_part_to_track` variable. The list of available options is in `body_part_detection.py`.
7. run the app with `python3 app.py`
8. do the exercise you want to do
9. stop the program by killing the window with the camera
10. a diagram should pop up like the ones above with the joint X and Y positions in time

## Future work
There's a lot that can be done to improve this application from different angles.

Here's a list of features I remember we got working but I can't seem to validate as I'm writing this:
- validate the peak and valley counting algorithm works iteratively and prints the counting in the terminal
- make the app work on the phone
- make it so that you don't have to specify the relevant joint that moves, there should be an algorithm that figures that out

Here's a list of future improvements:
- clean up the code
- count out loud as the exercise is being done
- detect if the form of the exercise is good (similarity of the amplitudes in time)
- detect which exercise the subject is doing
- count differently, instead of tracking the joint that makes the nicest peaks/valleys, count based on the exercise movement
