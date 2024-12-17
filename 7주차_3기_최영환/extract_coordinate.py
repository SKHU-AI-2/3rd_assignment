import cv2
import mediapipe as mp
import os
import csv
# numpy version 1.24.3

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)  # True == 이미지 모드, False == 비디오 모드

CSV_FILE = 'extract_coordinates.csv'
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'left_shoulder_X', 'left_shoulder_Y', 
        'left_wrist_X', 'left_wrist_Y',
        'left_knee_X', 'left_knee_Y',
        'Label'
    ])

path = 'data1/'
image_folders = [os.path.join(path, name) for name in os.listdir(path)\
                  if os.path.isdir(os.path.join(path, name))] # 경로 가져오기

# sit 0, stand 1
for label, image_folder in enumerate(image_folders):
    for img_name in sorted(os.listdir(image_folder), key=lambda x: int(os.path.splitext(x)[0])): # 오름차순 정렬
        img_path = os.path.join(image_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        # height, width, _ = image.shape
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            left_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_wrist = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            left_knee = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]

            # CSV
            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    left_shoulder.x, left_shoulder.y, 
                    left_wrist.x, left_wrist.y,
                    left_knee.x, left_knee.y, 
                    label
                ])




