import os
import time

import cv2
import mediapipe as mp
from utils.hand import Hand,HandSnapshot
from utils.utils import draw_hand_landmarks

__window_name__ = 'Static Pose Data COllect'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def main():
    hand_poses = []
    for pose in Hand.Pose:
        hand_poses.append(pose.name)

    hand_poses.remove('UNDEFINED')
    print(hand_poses)
    #params
    
    pose_name = 'DIAG_THUMB_SIDE' 
    path_to_data = os.path.join(os.getcwd(), "Static_Dataset\static_dataset_1")
    reset = Falseq
    delay_between_snapshots = 0.5
    stop_after = 60;
    save_images = False
    record = True

    #tracking snapshots
    last_snapshot = time.time()
    snapshot_index = 0
    training_pose = Hand.Pose[pose_name]
    
    # Remove previously recorded files 
    files = os.listdir(path_to_data)
    files = [f for f in files if f.startswith('snapshot_') and pose_name in f]
    files.sort()
    
    if reset:
        for file in files:
                os.remove(os.path.join(path_to_data, file))
    else:
        files = [f for f in files if f.endswith('.dat')]
        if len(files) > 0:
            snapshot_index = int(files[-1][9:13]) + 1

    # Webcam input
    capture = cv2.VideoCapture(0)
    with mp_hands.Hands(static_image_mode=False,
                        model_complexity=1,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:

        # Detect hand movement while the video capture is on
        while capture.isOpened():
            success, image = capture.read()
            image = cv2.flip(image, 1)

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Hand detection results
            results = hands.process(image)
            hand_detected = bool(results.multi_hand_landmarks)
            if hand_detected:
                    # Draw hand landmarks
                    draw_hand_landmarks(image, results.multi_hand_landmarks[0])

            # Snapshot every delay seconds
            if time.time() - last_snapshot > delay_between_snapshots:

                if hand_detected:

                    # Get the landmarks
                    landmarks = results.multi_hand_landmarks[0].landmark

                    # Save the data (an optionally the corresponding image)
                    if record:
                        hand = Hand(pose=training_pose)
                        snapshot = HandSnapshot(hand=hand)
                        current_file = f'snapshot_{snapshot_index:04}_pose-{hand.pose.value}-{hand.pose.name}'
                        output_path = os.path.join(path_to_data, current_file)
                        snapshot.save_landmarks_vector(landmarks, path=output_path)
                        if save_images:
                            snapshot.save_processed_image(image, path=output_path)
                        
                        saved_at = time.strftime('%H:%M:%S')
                        print(f'# Saved snapshot #{snapshot_index} for pose "{pose_name} :" ({saved_at})')
                        last_snapshot = time.time()
                        snapshot_index += 1

                    # Only basic information about the detection
                    else:
                        which_hand = results.multi_handedness[0].classification[0].label
                        confidence = int(100 * results.multi_handedness[0].classification[0].score)
                        print('Hand detected={} | Confidence={}%'.format(which_hand, confidence))

            # Display the image (stop if ESC key is pressed)
            cv2.imshow(__window_name__, image)
            if cv2.waitKey(5) & 0xFF == 27 or  snapshot_index >= stop_after:
                break

if __name__ == '__main__':
    main()