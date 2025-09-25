import cv2
import mediapipe as mp
import pickle

# Load trained model
model = pickle.load(open("gesture_model.pkl", "rb"))

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Gesture dictionary (map label → emoji + meaning)
gesture_to_meaning = {
    "Super": {"emoji": "👍", "meaning": "Super"},
    "OK": {"emoji": "👌", "meaning": "OK"},
    "LoveYou": {"emoji": "🤟", "meaning": "Love You"},
    "Stop": {"emoji": "✋", "meaning": "Stop"},
    "Fist": {"emoji": "👊", "meaning": "Bro/Fist bump"},
    "Victory": {"emoji": "✌", "meaning": "Victory/Peace"}
}

def extract_landmarks(hand_landmarks):
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return data

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                features = extract_landmarks(hand_landmarks)

                prediction = model.predict([features])[0]

                if prediction in gesture_to_meaning:
                    emoji = gesture_to_meaning[prediction]["emoji"]
                    meaning = gesture_to_meaning[prediction]["meaning"]

                    cv2.putText(frame, f"{emoji} {meaning}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()