import numpy as np
import pandas as pd
import os
import csv 
folder_path = r"D:\CS\Grad\Dataset\\excel 3"
excels = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.csv')]
row = []
for file in excels:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    
    # Exclude the first row (header) and calculate the mean for each column
    column_means = df.iloc[1:, :].mean(axis=0)
    row.append (column_means)
    print(f"Mean values for each column in {file}:\n{column_means}\n")
    
fields = ["left_wrist_x", "left_wrist_y", "left_wrist_z", "left_shoulder_x", "left_shoulder_y", "left_shoulder_z",
            "Left_Elbow_x", "Left_Elbow_y", "Left_Elbow_z","right_wrist_x", "right_wrist_y", "right_wrist_z", "right_shoulder_x", "right_shoulder_y", "right_shoulder_z",
            "right_Elbow_x", "right_Elbow_y", "right_Elbow_z","left_pinky_x","left_pinky_y","left_pinky_z","right_pinky_x","right_pinky_y","right_pinky_z","left_index_x","left_index_y","left_index_z","right_index_x","right_index_y","right_index_z","left_thumb_x","left_thumb_y","left_thumb_z","right_thumb_x","right_thumb_y","right_thumb_z","left_hip_x","left_hip_y","left_hip_z","right_hip_x","right_hip_y","right_hip_z","left_knee_x","left_knee_y","left_knee_z","right_knee_x","right_knee_y","right_knee_z","left_ankle_x","left_ankle_y","left_ankle_z","right_ankle_x","right_ankle_y","right_ankle_z","left_heel_x","left_heel_y","left_heel_z","right_heel_x","right_heel_y","right_heel_z","left_foot_index_x","left_foot_index_y","left_foot_index_z","right_foot_index_x","right_foot_index_y","right_foot_index_z"]


with open( "e3mean.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(row)
