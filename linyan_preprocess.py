# -*- coding:utf-8 -*-
"""
created on June 28, 2017

@author: Linyan_Wang
"""
import cv2
import rtpose
import csv
from numpy import *
import numpy as np
from sklearn import svm
import sys

def draw_2d_joints(image, result):
    people_num = len(result)
    for i in range(0,people_num):
        cv2.circle(image, (int(result[i][0]), int(result[i][1])), 6, (255, 0, 0), 3)
        cv2.circle(image, (int(result[i][2]), int(result[i][3])), 6, (255, 85, 0), 3)
        cv2.circle(image, (int(result[i][4]), int(result[i][5])), 6, (255, 170, 0), 3)
        cv2.circle(image, (int(result[i][6]), int(result[i][7])), 6, (255, 255, 0), 3)
        cv2.circle(image, (int(result[i][8]), int(result[i][9])), 6, (170, 255, 0), 3)
        cv2.circle(image, (int(result[i][10]), int(result[i][11])), 6, (85, 255, 0), 3)
        cv2.circle(image, (int(result[i][12]), int(result[i][13])), 6, (0, 255, 0), 3)
        cv2.circle(image, (int(result[i][14]), int(result[i][15])), 6, (0, 255, 85), 3)
        cv2.circle(image, (int(result[i][16]), int(result[i][17])), 6, (0, 255, 170), 3)
        cv2.circle(image, (int(result[i][18]), int(result[i][19])), 6, (0, 255, 255), 3)
        cv2.circle(image, (int(result[i][20]), int(result[i][21])), 6, (0, 170, 255), 3)
        cv2.circle(image, (int(result[i][22]), int(result[i][23])), 6, (0, 85, 255), 3)
        cv2.circle(image, (int(result[i][24]), int(result[i][25])), 6, (0, 0, 255), 3)
        cv2.circle(image, (int(result[i][26]), int(result[i][27])), 6, (85, 0, 255), 3)
        cv2.circle(image, (int(result[i][28]), int(result[i][29])), 6, (170, 0, 255), 3)
        cv2.circle(image, (int(result[i][30]), int(result[i][31])), 6, (255, 0, 255), 3)
        cv2.circle(image, (int(result[i][32]), int(result[i][33])), 6, (255, 0, 170), 3)
        cv2.circle(image, (int(result[i][34]), int(result[i][35])), 6, (255, 0, 85), 3)


video_folder = ['boxing','handclapping','handwaving','jogging','running','walking']
# video_folder = ['boxing','handclapping','handwaving','jogging','running','walking']
person_num = ['21','22','23','24','25']
# person_num = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']
d_num = ['d1','d2','d3','d4']
file_name = open("action_val.csv", "w")
writer = csv.writer(file_name)
# writer.writerow(['l_ear_x','l_ear_y','r_ear_x','r_ear_y','l_eye_x','l_eye_y','r_eye_x','r_eye_y','nose_x','nose_y','neck_x','neck_y','l_shoulder_x','l_shoulder_y','r_shoulder_x','r_shoulder_y','l_elbow_x','l_elbow_y','r_elbow_x','r_elbow_y','l_wrist_x','l_wrist_y','r_wrist_x','l_wrist_y','l_hip_x','l_hip_y','r_hip_x','r_hip_y','l_knee_x','l_knee_y','r_knee_x','r_knee_y','l_ankle_x','l_ankle_y','r_ankle_x','r_ankle_y','label_boxing','label_handclapping'])

a = []
for i in range(1806):
    a.append(i)

writer.writerow(a)

inter_diff = []
outer_np = []
pre_pose = []
cur_pose = []
cv2.namedWindow("action", flags=0)
# label = "None"
for ele in video_folder:
    for p in person_num:
        for d in d_num:
            count = 0
            video_path = 'KTH/' + ele + '/person' + p + '_' + ele + '_' + d + '_uncomp' + '.avi'
            print video_path
            rlist = []
            # col_num = [1]
            # rlist.append(col_num)
            cap = cv2.VideoCapture(video_path)
            print "get video"
            while (cap.isOpened()):
                ret, image = cap.read()
                if not ret:
                    break
                result = rtpose.rtpose(image)
                pre_pose = cur_pose
                cur_pose = result
                if len(result) == 0:
                    pass
                else:
                    draw_2d_joints(image, result)
                    # cv2.putText(image, label, (int(cur_pose[0][34]), int(cur_pose[0][35])), 0, 0.3, 0)
                    if count >= 1:
                        if len(pre_pose) == 0:
                            pass
                        else:
                            l_ear_x = int(cur_pose[0][34]) - int(pre_pose[0][34])
                            l_ear_y = int(cur_pose[0][35]) - int(pre_pose[0][35])
                            r_ear_x = int(cur_pose[0][32]) - int(pre_pose[0][32])
                            r_ear_y = int(cur_pose[0][33]) - int(pre_pose[0][33])
                            l_eye_x = int(cur_pose[0][30]) - int(pre_pose[0][30])
                            l_eye_y = int(cur_pose[0][31]) - int(pre_pose[0][31])
                            r_eye_x = int(cur_pose[0][28]) - int(pre_pose[0][28])
                            r_eye_y = int(cur_pose[0][29]) - int(pre_pose[0][29])
                            nose_x = int(cur_pose[0][0]) - int(pre_pose[0][0])
                            nose_y = int(cur_pose[0][1]) - int(pre_pose[0][1])
                            neck_x = int(cur_pose[0][2]) - int(pre_pose[0][2])
                            neck_y = int(cur_pose[0][3]) - int(pre_pose[0][3])
                            l_shoulder_x = int(cur_pose[0][10]) - int(pre_pose[0][10])
                            l_shoulder_y = int(cur_pose[0][11]) - int(pre_pose[0][11])
                            r_shoulder_x = int(cur_pose[0][8]) - int(pre_pose[0][8])
                            r_shoulder_y = int(cur_pose[0][9]) - int(pre_pose[0][9])
                            l_elbow_x = int(cur_pose[0][14]) - int(pre_pose[0][14])
                            l_elbow_y = int(cur_pose[0][15])- int(pre_pose[0][15])
                            r_elbow_x = int(cur_pose[0][12]) - int(pre_pose[0][12])
                            r_elbow_y = int(cur_pose[0][13]) - int(pre_pose[0][13])
                            l_wrist_x = int(cur_pose[0][18]) - int(pre_pose[0][18])
                            l_wrist_y = int(cur_pose[0][19]) - int(pre_pose[0][19])
                            r_wrist_x = int(cur_pose[0][16]) - int(pre_pose[0][16])
                            r_wrist_y = int(cur_pose[0][17]) - int(pre_pose[0][17])
                            l_hip_x = int(cur_pose[0][6]) - int(pre_pose[0][6])
                            l_hip_y = int(cur_pose[0][7])- int(pre_pose[0][7])
                            r_hip_x = int(cur_pose[0][4]) - int(pre_pose[0][4])
                            r_hip_y = int(cur_pose[0][5]) - int(pre_pose[0][5])
                            l_knee_x = int(cur_pose[0][22]) - int(pre_pose[0][22])
                            l_knee_y = int(cur_pose[0][23]) - int(pre_pose[0][23])
                            r_knee_x = int(cur_pose[0][20]) - int(pre_pose[0][20])
                            r_knee_y = int(cur_pose[0][21]) - int(pre_pose[0][21])
                            l_ankle_x = int(cur_pose[0][26]) - int(pre_pose[0][26])
                            l_ankle_y = int(cur_pose[0][27]) - int(pre_pose[0][27])
                            r_ankle_x = int(cur_pose[0][24]) - int(pre_pose[0][24])
                            r_ankle_y = int(cur_pose[0][25]) - int(pre_pose[0][25])
                            rlist.append(l_ear_x)
                            rlist.append(l_ear_y)
                            rlist.append(r_ear_x)
                            rlist.append(r_ear_y)
                            rlist.append(l_eye_x)
                            rlist.append(l_eye_y)
                            rlist.append(r_eye_x)
                            rlist.append(r_eye_y)
                            rlist.append(nose_x)
                            rlist.append(nose_y)
                            rlist.append(neck_x)
                            rlist.append(neck_y)
                            rlist.append(l_shoulder_x)
                            rlist.append(l_shoulder_y)
                            rlist.append(r_shoulder_x)
                            rlist.append(r_shoulder_y)
                            rlist.append(l_elbow_x)
                            rlist.append(l_elbow_y)
                            rlist.append(r_elbow_x)
                            rlist.append(r_elbow_y)
                            rlist.append(l_wrist_x)
                            rlist.append(l_wrist_y)
                            rlist.append(r_wrist_x)
                            rlist.append(r_wrist_y)
                            rlist.append(l_hip_x)
                            rlist.append(l_hip_y)
                            rlist.append(r_hip_x)
                            rlist.append(r_hip_y)
                            rlist.append(l_knee_x)
                            rlist.append(l_knee_y)
                            rlist.append(r_knee_x)
                            rlist.append(r_knee_y)
                            rlist.append(l_ankle_x)
                            rlist.append(l_ankle_y)
                            rlist.append(r_ankle_x)
                            rlist.append(r_ankle_y)
                            if len(rlist) == 1800:
                                if ele == 'boxing':
                                    rlist.append(1)
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(0)
                                elif ele == 'handclapping':
                                    rlist.append(0)
                                    rlist.append(1)
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(0)
                                elif ele == 'handwaving':
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(1)
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(0)
                                elif ele == 'jogging':
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(1)
                                    rlist.append(0)
                                    rlist.append(0)
                                elif ele == 'running':
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(1)
                                    rlist.append(0)
                                elif ele == 'walking':
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(0)
                                    rlist.append(1)
                                writer.writerow(rlist)
                                break

                    count += 1
                    print "frame:", count
                    # writer.writerow(rlist)
                    print video_path + "over"
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        # break;
                        sys.exit(0)
                cv2.imshow("action", image)
            cap.release()
            cv2.destroyAllWindows()

file_name.close()