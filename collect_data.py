import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
gesture_name = input("Enter gesture label (e.g., OK, Super, LoveYou, Stop): ")

def extract_landmarks(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

# Open CSV safely with 'with'
with open("gesture_data.csv", "a", newline="") as f:
    writer = csv.writer(f)
    sample_count = 0

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

                    # Convert all floats to strings to avoid CSV issues
                    writer.writerow([gesture_name] + [str(f) for f in features])
                    f.flush()  # immediately write to disk
                    sample_count += 1

            cv2.putText(frame, f'Samples: {sample_count}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Collecting Data - " + gesture_name, frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

cap.release()
cv2.destroyAllWindows()
print(f"Data collection stopped. Total samples saved: {sample_count}")