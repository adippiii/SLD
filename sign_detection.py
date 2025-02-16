import pickle

import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
from pynput.keyboard import Key,Controller

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

engine = pyttsx3.init()
engine.setProperty('rate', 85)

def detect_fingers():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(0)
    
    labels_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: ' '}

    letters = []
    word = []
    punctuation = []
    sentence = []

    keyboard = Controller()

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            try:
                prediction = model.predict([np.asarray(data_aux)])
            except:
                pass

            predicted_character = labels_dict[int(prediction[0])]

            try:
                if predicted_character not in labels_dict:
                    letters.append(predicted_character)
            except:
                letters.append(" ")

                
            if len(letters) == 18:
                letter = letters[len(letters)-1]
                keyboard.type(letter)
                if predicted_character != ' ':
                    word.append(letter)
                else:
                    res = ''.join(word)
                    engine.say(res)
                    engine.runAndWait()
                    print(res)
                    word = []

                    sentence.append(res)

                    punctuation.append(' ')
                    if len(punctuation) == 2:
                        res = ' '.join(sentence)
                        engine.say(res)
                        engine.runAndWait()
                        print(res)
                        sentence = []
                        punctuation = []
                letters = []

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow("Finger Detection", frame)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_fingers()
