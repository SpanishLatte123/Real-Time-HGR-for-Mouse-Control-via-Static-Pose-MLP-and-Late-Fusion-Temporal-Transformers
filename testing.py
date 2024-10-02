import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from collections import Counter

def find_dominant_item(arr):
    # Count occurrences of each item using Counter
    counts = Counter(arr)
    
    # Find the item with the highest count
    dominant_item, count = counts.most_common(1)[0]
    
    return count,dominant_item

actions = ['CLICK1','CLICK2','THROW_UP','THROW_DOWN','THROW_RIGHT','THROW_LEFT','OPEN_TWICE','D_CLICK2','D_CLICK2','ZOOM_IN','ZOOM_OUT','POINT_1','POINT2']
seq_length = 65

model = load_model('Trained Models/dynamic/1_temporal_transformer_quad_noD0X.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

seq = []
action_seq = []

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            #print(res)
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
            #print('JOINTS:',joint)
            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # [20, 3]
            #print('V1:',v1)
            #print('V2:',v2)
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            #print('Normalized V',v)
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)
            #print('Seq',seq)
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]
            print(conf)

            #if confidence is less than 95% dont consider guess
            if conf < .85:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 30:
                continue

            this_action = '-'
            latest =[]

            #get the last 10 frames and append to latest array
            for i in range(20):
                latest.append(action_seq[0-i])

            #return the count and the most dominant item in the latest array
            countz,final_pred = find_dominant_item(latest)
            print(latest)
            print(final_pred,'-',countz)
            #if the last 2 items are the same to the most dominant item count is more than 8 
            if(countz > 10 and action_seq[-1] == action_seq[-2] == final_pred):
                this_action = action

            # Calculate the position of the text in the upper right-hand corner
            text = f'{this_action.upper()}'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
            text_x = img.shape[1] - text_size[0] - 10  
            text_y = text_size[1] + 10  

            cv2.putText(img, text, org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 0), thickness=2)
            time_steps = np.arange(0, 10, 0.1)
    
            
    # out.write(img0)
    # out2.write(img)
    cv2.imshow('DYNAMIC GESTURE TESTER', img)
    if cv2.waitKey(1) == ord('q'):
        break