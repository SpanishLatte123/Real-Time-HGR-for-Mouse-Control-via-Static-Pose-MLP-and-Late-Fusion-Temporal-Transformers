import cv2
import mediapipe as mp
import numpy as np
import threading
import pyautogui
from keras.models import load_model
from collections import Counter

from utils.hand import Hand
from utils.mouse_controller import HgrMouseControl
from utils.utils import draw_hand_landmarks, draw_palm_center, draw_bounds, draw_scrolling_origin
from utils.utils import write_pose
from utils.gestureTracker import Gesturer
from utils.macros_controller import MacroGestureControl

# Load models
static_model = load_model("Trained Models/best_models/5_static_model.h5")
dynamic_model = load_model("Trained Models/best_models/10_macros_transformers_v2.h5")

# Helper functions
def find_dominant(arr):
    counts = Counter(arr)
    dominant_item, count = counts.most_common(1)[0]
    return count, dominant_item

def get_dynamic_features(res):
    joint = np.zeros((21, 4))
    for j, lm in enumerate(res.landmark):
        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
    v = v2 - v1
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
    angle = np.arccos(np.einsum('nt,nt->n',
                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
    angle = np.degrees(angle)
    features = np.concatenate([joint.flatten(), angle])
    return features

def predict_gesture(input_data):
    global predict_thread, predict_lock, action_seq, actions, this_action
    with predict_lock:
        y_pred = dynamic_model.predict(input_data).squeeze()
        i_pred = int(np.argmax(y_pred))
        conf = y_pred[i_pred]

        # Activation rules
        if conf >= 0.80:
            action = actions[i_pred]
            action_seq.append(action)
            if len(action_seq) >= 10:
                latest = [action_seq[-i] for i in range(1, 11)]
                countz, final_pred = find_dominant(latest)
                if countz > 5 and action_seq[-1] == action_seq[-2] == final_pred:
                    this_action = action

def perform_dynamic_gesture_thread(gesturer):
    gesture_controller.perform_dynamic_gesture(gesturer)

# Mediapipe Instantiation
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Disable Fail safe of PyautoGUI
pyautogui.FAILSAFE = False

# OpenCV Params
scale = 0.4
height = int(1080 * scale)
width = int(1920 * scale)
capture = cv2.VideoCapture(0)
show_vid = True

# Mouse Params
hand_controller = HgrMouseControl(sensitivity=0.5, margin=0.25, scrolling_threshold=0.1, scrolling_speed=40, min_cutoff_filter=0.01, beta_filter=10)
gesture_controller = MacroGestureControl()
gesturer_1 = Gesturer()

# Threading params
predict_thread = None
predict_lock = threading.Lock()
gesture_thread = None
gesture_lock = threading.Lock()

# Dynamic gestures vars
seq = []
seq_length = 30
action_seq = []
actions = ['NO_GESTURES', 'NO_GESTURES', 'CLICK1', 'CLICK2', '3_SWIPE_LEFT', '3_SWIPE_RIGHT', 'TWIRL_OK', 'TWIRL_OK_HORIZONTAL', 'FINGER_HEART', 'THUMBS_UP_DIAG']

with mp_hands.Hands(static_image_mode=False, model_complexity=1, max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.5) as hands:
    while capture.isOpened():
        success, image = capture.read()
        image = cv2.flip(image, 1)
        image = cv2.resize(image, (width, height))

        boundary = hand_controller.action_area(image)
        draw_bounds(image, boundary)
        
        results = hands.process(image)
        hand_detected = bool(results.multi_hand_landmarks)

        if not success:
            print("No Camera Found")
            continue
        
        if hand_detected:
            hand = Hand()
            
            # Process static vectors
            landmarks = results.multi_hand_landmarks[0].landmark
            raw_landmark_vector = hand.vectorize_landmarks(landmarks)
            palm_center = hand_controller.get_palm_center(raw_landmark_vector)

            # Predict static poses on main thread
            processed_landmark_vector = hand.feature_process_landmarks(raw_landmark_vector)
            static_probabilities = static_model.predict(processed_landmark_vector, verbose=0).flatten()
            static_confidence = np.max(static_probabilities)
            static_predicted_pose = np.argmax(static_probabilities)
            hand.pose = Hand.Pose(static_predicted_pose)

            # Get features on main thread
            res = results.multi_hand_landmarks[0]
            dynamic_features = get_dynamic_features(res)
            seq.append(dynamic_features)

            # Display Hand Landmarks
            draw_hand_landmarks(image, res, hand.pose.name)

            # Do dynamic predictions on separate thread
            if len(seq) >= seq_length:
                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                if predict_thread is None or not predict_thread.is_alive():
                    predict_thread = threading.Thread(target=predict_gesture, args=(input_data,))
                    predict_thread.start()
            
            # Call mouse controller
            if gesture_controller.current_state == MacroGestureControl.GestureState.STANDARD:
                hand_controller.operate_mouse(hand, palm_center, static_confidence, min_confidence=0.8)

        if show_vid:
            if hand_detected:
                draw_palm_center(image, palm_center, size=0.03)
                write_pose(image, this_action, gesture_controller.current_state)
                predicted_gesture = actions.index(this_action)
                gesturer_1.Gesturez = Gesturer.Gesturez(predicted_gesture)
                
                # Perform dynamic gesture on separate thread
                if gesture_thread is None or not gesture_thread.is_alive():
                    gesture_thread = threading.Thread(target=perform_dynamic_gesture_thread, args=(gesturer_1,))
                    gesture_thread.start()
                
            else:
                this_action = 'NO_GESTURES'
                write_pose(image, this_action, gesture_controller.current_state)

            # Draw Scrolling Boundaries
            if hand_controller.current_mouse_state == HgrMouseControl.MouseState.SCROLLING:
                draw_scrolling_origin(image, hand_controller.scrolling_origin, hand_controller.scrolling_threshold)

            cv2.namedWindow("HGR MOUSE V1", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("HGR MOUSE V1", cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow("HGR MOUSE V1", image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

capture.release()
