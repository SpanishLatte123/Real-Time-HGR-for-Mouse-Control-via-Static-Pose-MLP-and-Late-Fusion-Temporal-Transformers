import random

import numpy
import cv2
import mediapipe as mp
from .macros_controller import MacroGestureControl

# Mediapipe shortcuts
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def clamp(value, min_value, max_value):
    value = max(value, min_value)
    value = min(value, max_value)
    return value

def draw_hand_landmarks(image, hand_landmark,pose=""):
    # Extract hand landmarks
    landmarks = hand_landmark.landmark

    # Convert landmarks to pixel coordinates
    image_height, image_width, _ = image.shape
    landmarks_pixel = []
    for landmark in landmarks:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        landmarks_pixel.append((x, y))

    # Draw hand landmarks and connections
    mp_drawing.draw_landmarks(image,
                              hand_landmark,
                              mp_hands.HAND_CONNECTIONS,
                              mp_drawing_styles.get_default_hand_landmarks_style(),
                              mp_drawing_styles.get_default_hand_connections_style())

    # Calculate bounding box coordinates with margins
    min_x = min(landmarks_pixel, key=lambda x: x[0])[0]
    min_y = min(landmarks_pixel, key=lambda x: x[1])[1]
    max_x = max(landmarks_pixel, key=lambda x: x[0])[0]
    max_y = max(landmarks_pixel, key=lambda x: x[1])[1]

    margin_x = int((max_x - min_x) * 0.15)
    margin_y = int((max_y - min_y) * 0.15)

    min_x -= margin_x
    min_y -= margin_y
    max_x += margin_x
    max_y += margin_y

    # Draw bounding box around the hand with margins
    cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 255, 0), 2)

    # Add text above the bounding box
    text = f"Static Pose: {pose}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_position = ((min_x + max_x - text_size[0]) // 2, min_y - 10)
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

def draw_palm_center(image, palm_center, color_BGR=(255,255,255), size=0.05):
    height, width, _ = image.shape
    size = int((size * height) / 2)
    # size = size // 2
    cursor_xy = (palm_center * numpy.array([width, height])).astype(int)
    cv2.rectangle(image, tuple(cursor_xy - size), 
                        tuple(cursor_xy + size),
                        color_BGR, thickness=-1)

def draw_bounds(image, bounds):
    # Bounds
    height, width, _ = image.shape
    xmin, ymin, xmax, ymax = bounds

    # Draw action
    hidden = numpy.zeros_like(image, numpy.uint8)
    alpha = 0.30
    color = (1,1,1)
    cv2.rectangle(hidden, (0, 0), (xmin, height), color, -1)
    cv2.rectangle(hidden, (xmin, 0), (xmax, ymin), color, -1)
    cv2.rectangle(hidden, (xmax, 0), (width, height), color, -1)
    cv2.rectangle(hidden, (xmin, ymax), (xmax, height), color, -1)
    mask = hidden.astype(bool)
    image[mask] = cv2.addWeighted(image, alpha, hidden, 1-alpha, 0)[mask]

def draw_scrolling_origin(image, origin, threshold, color_BGR=(255,255,0), line_thickness=1, text_margin=(10,10)):
    height, width, _ = image.shape
    bottom = int( (origin + threshold) * height )
    top = int( (origin - threshold) * height )
    start_bottom = (0, bottom)
    end_bottom = (width, bottom)
    start_top = (0, top)
    end_top = (width, top)
    # Draw line
    cv2.line(image, start_top, end_top, color_BGR, line_thickness)
    cv2.line(image, start_bottom, end_bottom, color_BGR, line_thickness)
    # Write helper text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.75
    v_offset = int(0.075 * font_size * height)
    cv2.putText(image, 'Scroll up', (text_margin[0], top-text_margin[1]), font, font_size, color_BGR, thickness=2)
    cv2.putText(image, 'Scroll down', (text_margin[0], bottom+v_offset), font, font_size, color_BGR, thickness=2)

def write_pose(image, pose='TBA', state=MacroGestureControl.GestureState.STANDARD, color_BGR=(255, 255, 0), thickness=2, font_size=1, margin=(5, 10)):
    height, _, width = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(image, f'Dynamic Gesture: {pose}', (margin[0], height - margin[1]), font, font_size, color_BGR, thickness=thickness)
    if state == MacroGestureControl.GestureState.STANDARD:
        cv2.putText(image, f'ON', (margin[0], margin[1] + 25), font, font_size, (0, 255, 0), thickness=thickness)
    else:
        cv2.putText(image, f'OFF', (margin[0], margin[1] + 25), font, font_size, (0, 0, 255), thickness=thickness)

    '''# Overlay the image in the upper right corner
    overlay_image = cv2.imread('Figures/dsp.png')
    overlay_image = cv2.resize(overlay_image, (130, 65))
    # Overlay the image in the upper right corner with opacity
    if overlay_image is not None:
        overlay_height, overlay_width = overlay_image.shape[:2]
        overlay_x = width - overlay_width - margin[0]
        overlay_y = height - 65

        # Create an overlay for the image with opacity
        alpha_overlay = 0.45
        beta = 1.0 - alpha_overlay
        image[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width] = cv2.addWeighted(
            image[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width], 
            beta, 
            overlay_image, 
            alpha_overlay, 
            0
        )

    return image'''


def train_test_split(X, y, test_size=0.0, seed=None):
    # Seed
    random.seed(seed)
    # Bounds
    n_samples = X.shape[0]
    n_samples_train = int( (1.0 - test_size) * n_samples )
    # Separate datasets
    indices_train = random.sample(range(n_samples), n_samples_train)
    indices_test = [i for i in range(n_samples) if i not in indices_train]
    X_train = X[indices_train, :]
    X_test = X[indices_test, :]
    y_train = y[indices_train]
    y_test = y[indices_test]
    return X_train, X_test, y_train, y_test