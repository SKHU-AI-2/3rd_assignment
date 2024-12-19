import cv2
import mediapipe as mp
import os
import csv
# numpy version 1.24.3


image_folder = 'data/sit/'
csv_file = 'extract_coordinates_sit.csv'


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)  # True == 이미지 모드, False == 비디오 모드


with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'Image', 
        'left shoulder X', 'left shoulder Y', 
        'left wrist X', 'left wrist Y', 
        'right shoulder X', 'right shoulder Y',
        'right wrist X', 'right wrist Y', 
        'Label'
    ])


for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue

    height, width, _ = image.shape

    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    
    if result.pose_landmarks:
        left_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_wrist = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_wrist = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # 절대 좌표
        left_shoulder_x = int(left_shoulder.x * width)
        left_shoulder_y = int(left_shoulder.y * height)
        left_wrist_x = int(left_wrist.x * width)
        left_wrist_y = int(left_wrist.y * height)
        right_shoulder_x = int(right_shoulder.x * width)
        right_shoulder_y = int(right_shoulder.y * height)
        right_wrist_x = int(right_wrist.x * width)
        right_wrist_y = int(right_wrist.y * height)
        # CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                img_name, 
                left_shoulder_x, left_shoulder_y, 
                left_wrist_x, left_wrist_y, 
                right_shoulder_x, right_shoulder_y,
                right_wrist_x, right_wrist_y,
                'sit'
            ])