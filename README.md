# workout-counter
Counting reps in the gym/home.

## idea
The goal of this project is to create an application that will be able to count exercise repetitions while recording a person during a workout.

## realisation ideas
- [x] detect_body_parts(frame) -> body_part_positions
- [ ] trim_video_to_exercise(body_part_positions) -> start, end
- [ ] classify the exercise(body_part_positions) -> exercise
- [ ] detect_relevant_body_part(body_part_positions) -> body_part
- [ ] detect_repetition(body_part) -> number_of_repetitions
  - [x] smoothen(X) -> X
  - [x] find_peaks(X) -> peaksX
  - [x] peak_prominences(X) -> prominencesX
  - [ ] epsilon = minimum movement amount relative to body_part length
  - [x] count(peaksX, prominencesX, epsilon) -> number_of_repetitions
- [ ] create an iterative approach

## detecting human positions
The MoveNet neural network is used to detect body parts in workouts.

<img src="./img/body_part_detection.png" width=200>

## drawing relevant point movement
![pullups](./img/pullups.png)
![situps](./img/situps.png)
![bicep curls](./img/bicep_curls.png)
![tricep extensions](./img/tricep_extensions.png)
