

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

import cv2
import rtpose
from numpy import *

dim_hidden = 512
learning_rate = 0.001
video_frames = 25
step_frame = 25
time_step = video_frames
pose_num = 36
batch_size = 1
stddev = 0.01
class_num = 6
correct_1 = 0
wrong_1 = 0
correct_2 = 0
wrong_2 = 0
acc_boxing = []
acc_handclapping = []
inter_diff = []
outer_np = []
pre_pose = []
cur_pose = []
step_logit = []
rlist = []
draw_label = "None"

def draw_2d_joints(image, result):
    people_num = len(result)
    for i in range(0,people_num):
        cv2.circle(image, (int(result[i][0]), int(result[i][1])), 3, (255, 0, 0), 3)
        cv2.circle(image, (int(result[i][2]), int(result[i][3])), 3, (255, 85, 0), 3)
        cv2.circle(image, (int(result[i][4]), int(result[i][5])), 3, (255, 170, 0), 3)
        cv2.circle(image, (int(result[i][6]), int(result[i][7])), 3, (255, 255, 0), 3)
        cv2.circle(image, (int(result[i][8]), int(result[i][9])), 3, (170, 255, 0), 3)
        cv2.circle(image, (int(result[i][10]), int(result[i][11])), 3, (85, 255, 0), 3)
        cv2.circle(image, (int(result[i][12]), int(result[i][13])), 3, (0, 255, 0), 3)
        cv2.circle(image, (int(result[i][14]), int(result[i][15])), 3, (0, 255, 85), 3)
        cv2.circle(image, (int(result[i][16]), int(result[i][17])), 3, (0, 255, 170), 3)
        cv2.circle(image, (int(result[i][18]), int(result[i][19])), 3, (0, 255, 255), 3)
        cv2.circle(image, (int(result[i][20]), int(result[i][21])), 3, (0, 170, 255), 3)
        cv2.circle(image, (int(result[i][22]), int(result[i][23])), 3, (0, 85, 255), 3)
        cv2.circle(image, (int(result[i][24]), int(result[i][25])), 3, (0, 0, 255), 3)
        cv2.circle(image, (int(result[i][26]), int(result[i][27])), 3, (85, 0, 255), 3)
        cv2.circle(image, (int(result[i][28]), int(result[i][29])), 3, (170, 0, 255), 3)
        cv2.circle(image, (int(result[i][30]), int(result[i][31])), 3, (255, 0, 255), 3)
        cv2.circle(image, (int(result[i][32]), int(result[i][33])), 3, (255, 0, 170), 3)
        cv2.circle(image, (int(result[i][34]), int(result[i][35])), 3, (255, 0, 85), 3)



# f = open('action_train.csv')
# df = pd.read_csv(f)
# df = np.array(df)
# train_x = df[0:,0:10800]
# train_x = np.reshape(train_x, [-1,video_frames, pose_num])
#
# train_y = df[0:,10800:10802]
# train_y = np.tile(train_y, 30)
# print(np.shape(train_y))
# train_y = np.reshape(train_y,[-1,class_num])
#
video = tf.placeholder(tf.float32, shape=(batch_size, video_frames, pose_num))
# label = tf.placeholder(tf.int32, shape=(batch_size, class_num))
#
initializer = tf.random_normal_initializer(stddev=stddev)
#
dec_w = tf.get_variable("w", [dim_hidden, class_num], initializer = initializer)
dec_b = tf.get_variable("b", class_num, initializer = tf.constant_initializer(0))
#
lstm = tf.contrib.rnn.LSTMCell(dim_hidden, initializer=tf.random_normal_initializer(stddev=0.03), use_peepholes=True, state_is_tuple=True)
#
state = lstm.zero_state(batch_size,tf.float32)
# initial_state =  state = tf.zeros([batch_size, lstm.state_size])
#
for idx in range(time_step):
    input = video[:,idx,:]
    input = tf.reshape(input, [-1,pose_num])

    with tf.variable_scope("lstm"):
        output, state = lstm(input, state)
    tf.get_variable_scope().reuse_variables()
logit = tf.nn.xw_plus_b(output, dec_w, dec_b)
s_logit = tf.nn.softmax(logit)
#
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit)
# tf.get_variable_scope().reuse_variables()
#
# loss = tf.reduce_mean(cross_entropy)
# # optimizer = tf.train.AdamOptimizer(learning_rate)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# train_op = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('./lstm_model')
saver.restore(sess, './lstm_model/pose.model')

# img_w = 1920
# img_h = 1080
img_w = 400
img_h = 640
writer = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc(*'MJPG'),20,(img_w,img_h),1)
# writer = cv2.VideoWriter('out.avi', cv2.cv.CV_FOURCC('M','J','P','G'),20,(img_w,img_h),1)

cv2.namedWindow("action", flags=0)

video_folder = ['running']
# video_folder = ['boxing','handclapping','handwaving','jogging','running','walking']
person_num = ['22']
# person_num = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']
d_num = ['d1']
# d_num = ['d1','d2','d3','d4']
for ele in video_folder:
    for p in person_num:
        for d in d_num:
            count = 0
            # video_path = 'KTH/' + ele + '/person' + p + '_' + ele + '_' + d + '_uncomp' + '.avi'
            video_path = './IMG_2873.m4v'
            print video_path
            cap = cv2.VideoCapture(video_path)
            print "get video"
            while (cap.isOpened()):
                ret, image = cap.read()
                image = cv2.resize(image, (400, 640), cv2.INTER_AREA)
                # if not ret:
                #     break
                result = rtpose.rtpose(image)
                pre_pose = cur_pose
                cur_pose = result
                if len(result) == 0:
                    draw_label = draw_label
                else:
                    draw_2d_joints(image, result)
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
                            l_elbow_y = int(cur_pose[0][15]) - int(pre_pose[0][15])
                            r_elbow_x = int(cur_pose[0][12]) - int(pre_pose[0][12])
                            r_elbow_y = int(cur_pose[0][13]) - int(pre_pose[0][13])
                            l_wrist_x = int(cur_pose[0][18]) - int(pre_pose[0][18])
                            l_wrist_y = int(cur_pose[0][19]) - int(pre_pose[0][19])
                            r_wrist_x = int(cur_pose[0][16]) - int(pre_pose[0][16])
                            r_wrist_y = int(cur_pose[0][17]) - int(pre_pose[0][17])
                            l_hip_x = int(cur_pose[0][6]) - int(pre_pose[0][6])
                            l_hip_y = int(cur_pose[0][7]) - int(pre_pose[0][7])
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
                            print "the length of rlist:", len(rlist)
                            if len(rlist) > 600 and len(rlist) % 900 == 0:

                                rlist = np.array(rlist)
                                rlist = rlist[np.newaxis, :]
                                rlist = np.reshape(rlist,[1,25,36])
                                [logit_] = sess.run([s_logit], feed_dict={video:rlist})

                                print logit_
                                indexes = np.argmax(logit_, axis=1)
                                for max_index in indexes:
                                    if max_index == 0:
                                        draw_label = "boxing"
                                    elif max_index == 1:
                                        draw_label = "handclapping"
                                    elif max_index == 2:
                                        draw_label = "handwaving"
                                    elif max_index == 3:
                                        draw_label = "jogging"
                                    elif max_index == 4:
                                        draw_label = "running"
                                    elif max_index == 5:
                                        draw_label = "walking"
                                    rlist = []
                                # if logit_[0][0] > logit_[0][1]:
                                #     draw_label = "boxing"
                                # else:
                                #     draw_label = "handclapping"
                                # rlist = []
                            else:
                                draw_label = draw_label

                    cv2.putText(image, draw_label, (int(result[0][2] - 50), int(result[0][3] - 50)), 0, 1,
                                [255, 255, 0], 4)
                if count % 25 == 0:
                    cv2.putText(image, str(count), (20, 20), 0, 1, [255, 0, 0], 4)
                else:
                    cv2.putText(image, str(count), (20, 20), 0, 1, [255, 255, 0], 4)
                cv2.imshow("action", image)
                cv2.waitKey(1)
                writer.write(image)
                print video_path + "over"
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break;
                count += 1
                print "frame:", count
            cap.release()
            writer.release()
            cv2.destroyAllWindows()



print 'test finish'

