import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from collections import deque

# ---------------- SETUP ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

cooldown_vertical = 0.8
last_vertical_time = 0

calibration_time = 3
start_time = time.time()

baseline_shoulder_x = None
baseline_hip_y = None

hip_history = deque(maxlen=5)

lr_state = "neutral"

print("Stand straight for calibration...")

# ---------------- LOOP ----------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    h, w, _ = img.shape

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        shoulder_center_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
        hip_y = int((left_hip.y + right_hip.y) / 2 * h)

        left_shoulder_y = int(left_shoulder.y * h)
        right_shoulder_y = int(right_shoulder.y * h)
        left_wrist_y = int(left_wrist.y * h)
        right_wrist_y = int(right_wrist.y * h)

        hip_history.append(hip_y)
        smoothed_hip_y = int(np.mean(hip_history))

        current_time = time.time()

        # ---------------- CALIBRATION ----------------
        if current_time - start_time < calibration_time:
            cv2.putText(img, "CALIBRATING...",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 3)

            baseline_shoulder_x = shoulder_center_x
            baseline_hip_y = smoothed_hip_y

        else:
            cv2.putText(img, "READY",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 3)

            # ---------- LEFT / RIGHT ----------
            threshold = 80
            neutral_zone = 40

            if lr_state == "neutral":

                if shoulder_center_x < baseline_shoulder_x - threshold:
                    pyautogui.press("left")
                    print("LEFT")
                    lr_state = "left"

                elif shoulder_center_x > baseline_shoulder_x + threshold:
                    pyautogui.press("right")
                    print("RIGHT")
                    lr_state = "right"

            if (baseline_shoulder_x - neutral_zone <
                shoulder_center_x <
                baseline_shoulder_x + neutral_zone):

                lr_state = "neutral"

            # ---------------- JUMP (HANDS UP) ----------------
            if (left_wrist_y < left_shoulder_y and
                right_wrist_y < right_shoulder_y and
                current_time - last_vertical_time > cooldown_vertical):

                pyautogui.press("up")
                print("JUMP (HANDS UP)")
                last_vertical_time = current_time

            # ---------------- CROUCH ----------------
            elif (smoothed_hip_y > baseline_hip_y + 110 and
                  current_time - last_vertical_time > cooldown_vertical):

                pyautogui.press("down")
                print("CROUCH")
                last_vertical_time = current_time

        mp_draw.draw_landmarks(img, results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Gesture Control - Subway Surfers", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()