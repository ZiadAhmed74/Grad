import cv2
import mediapipe as mp
import numpy as np
import csv
import os

videos = []
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_tracking_confidence=0.6)

folder_path = r"D:\CS\Grad\Dataset\\exro"
videos = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.MOV')]

for vid in videos:
    video_path = os.path.join(folder_path, vid)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {vid}")
        continue

    rows = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.resize(img, (500, 800))
        results = pose.process(img)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            h, w, c = img.shape
            opImg = np.zeros([h, w, c])
            opImg.fill(255)
            mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Extracted Poses", opImg)
            landmarks = results.pose_landmarks.landmark
            cv2.imshow("pose estimation", img)

            row = [
                str(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x),
                str(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y),
                str(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z),
                str(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x),
                str(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
                str(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z),
                str(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x),
                str(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y),
                str(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z),
                str(landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x),
                str(landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y),
                str(landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].z),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].z),
                str(landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x),
                str(landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y),
                str(landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].z),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].z),
                str(landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x),
                str(landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y),
                str(landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].z),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].z),
                str(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x),
                str(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
                str(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z),
                str(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x),
                str(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y),
                str(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z),
                str(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x),
                str(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y),
                str(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z),
                str(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x),
                str(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y),
                str(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z),
                str(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x),
                str(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y),
                str(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y),
                str(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z)
             ]
            

            rows.append(row)

            print(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, "Left Wrist x")
            print(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, "Left Wrist y")
            print(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z, "Left Wrist z")
            print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, "Left Shoulder x")
            print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, "Left Shoulder y")
            print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z, "Left Shoulder z")
            print(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, "Left Elbow x")
            print(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, "Left Elbow y")
            print(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z, "Left Elbow z")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, "Right Wrist x")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, "Right Wrist y")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z, "Right Wrist z")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, "Right Shoulder x")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, "Right Shoulder y")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z, "Right Shoulder z")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, "Right Elbow x")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, "Right Elbow y")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z, "Right Elbow z")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x, "Right Pinky x")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y, "Right Pinky y")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].z, "Right Pinky z")
            print(landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x, "Left Pinky x")
            print(landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y, "Left Pinky y")
            print(landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].z, "Left Pinky z")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x, "Right Index x")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y, "Right Index y")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].z, "Right Index z")
            print(landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x, "Left Index x")
            print(landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y, "Left Index y")
            print(landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].z, "Left Index z")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x, "Right Thumb x")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y, "Right Thumb y")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].z, "Right Thumb z")
            print(landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x, "Left Thumb x")
            print(landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y, "Left Thumb y")
            print(landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].z, "Left Thumb z")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, "Right Hip x")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, "Right Hip y")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z, "Right Hip z")
            print(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, "Left Hip x")
            print(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, "Left Hip y")
            print(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z, "Left Hip z")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, "Right Knee x")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, "Right Knee y")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z, "Right Knee z")
            print(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, "Left Knee x")
            print(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, "Left Knee y")
            print(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z, "Left Knee z")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, "Right Ankle x")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, "Right Ankle y")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z, "Right Ankle z")
            print(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, "Left Ankle x")
            print(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, "Left Ankle y")
            print(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z, "Left Ankle z")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, "Right Heel x")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y, "Right Heel y")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z, "Right Heel z")
            print(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, "Left Heel x")
            print(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y, "Left Heel y")
            print(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z, "Left Heel z")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, "Right Foot Index x")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y, "Right Foot Index y")
            print(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z, "Right Foot Index z")
            print(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, "Left Foot Index x")
            print(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y, "Left Foot Index y")
            print(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z, "Left Foot Index z")
            
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # field names
    fields = ["left_wrist_x", "left_wrist_y", "left_wrist_z", "left_shoulder_x", "left_shoulder_y", "left_shoulder_z",
              "Left_Elbow_x", "Left_Elbow_y", "Left_Elbow_z","right_wrist_x", "right_wrist_y", "right_wrist_z", "right_shoulder_x", "right_shoulder_y", "right_shoulder_z",
              "right_Elbow_x", "right_Elbow_y", "right_Elbow_z","left_pinky_x","left_pinky_y","left_pinky_z","right_pinky_x","right_pinky_y","right_pinky_z","left_index_x","left_index_y","left_index_z","right_index_x","right_index_y","right_index_z","left_thumb_x","left_thumb_y","left_thumb_z","right_thumb_x","right_thumb_y","right_thumb_z","left_hip_x","left_hip_y","left_hip_z","right_hip_x","right_hip_y","right_hip_z","left_knee_x","left_knee_y","left_knee_z","right_knee_x","right_knee_y","right_knee_z","left_ankle_x","left_ankle_y","left_ankle_z","right_ankle_x","right_ankle_y","right_ankle_z","left_heel_x","left_heel_y","left_heel_z","right_heel_x","right_heel_y","right_heel_z","left_foot_index_x","left_foot_index_y","left_foot_index_z","right_foot_index_x","right_foot_index_y","right_foot_index_z"]

    
    filename = vid + ".csv"
    filename = filename.replace(".MOV", "")
    os.chdir("D:\CS\Grad\Dataset\\excel 3")
    # writing to csv file
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

    cap.release()
    cv2.destroyAllWindows()
    print("Row appended successfully.")
