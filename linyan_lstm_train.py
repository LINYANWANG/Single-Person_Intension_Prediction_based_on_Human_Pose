

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

import cv2
import rtpose
from numpy import *

dim_hidden = 512
learning_rate = 0.0001
video_frames = 25
video_num_train = 0
video_num_val = 0
time_step = 25
repeat_times = 2
pose_num = 36
batch_size = 60
stddev = 0.01
class_num = 6
step_logit = []
sum_id =[]
val_id = []

fl_train = open('loss_train.txt', 'w')
fl_val = open('loss_val.txt', 'w')
fp_train = open('acc_train.txt', 'w')
fp_val = open('acc_val.txt', 'w')

f_t = open('action_train.csv')
f_v = open('action_val.csv')

# Train dataset
df_t = pd.read_csv(f_t)
df_t = np.array(df_t)

train_x = df_t[0:,0:1800]
video_num_train = len(train_x)
train_x = np.reshape(train_x, [-1, video_frames, pose_num])

train_y = df_t[0:,1800:1806]
train_y = np.tile(train_y, repeat_times)
train_y = np.reshape(train_y,[-1,class_num])

# Validation dataset
df_v = pd.read_csv(f_v)
df_v = np.array(df_v)

val_x = df_v[0:,0:1800]
video_num_val = len(val_x)
val_x = np.reshape(val_x, [-1,video_frames, pose_num])

val_y = df_v[0:,1800:1806]
val_y = np.tile(val_y, repeat_times)
val_y = np.reshape(val_y,[-1,class_num])

video = tf.placeholder(tf.float32, shape=(batch_size, video_frames, pose_num))
label = tf.placeholder(tf.int32, shape=(batch_size, class_num))

initializer = tf.random_normal_initializer(stddev=stddev)

dec_w = tf.get_variable("w", [dim_hidden, class_num], initializer = initializer)
dec_b = tf.get_variable("b", class_num, initializer = tf.constant_initializer(0))

lstm = tf.contrib.rnn.LSTMCell(dim_hidden, initializer=tf.random_normal_initializer(stddev=0.03), use_peepholes=True, state_is_tuple=False)
# multi_lstm = tf.contrib.rnn.MultiRNNCell([basic_lstm] * 2, state_is_tuple = False)
# lstm = tf.contrib.rnn.AttentionCellWrapper(multi_lstm, 2, 512)

state = lstm.zero_state(batch_size,tf.float32)
# initial_state =  state = tf.zeros([batch_size, lstm.state_size])

for idx in range(time_step):
    input = video[:,idx,:]
    input = tf.reshape(input, [-1,pose_num])

    with tf.variable_scope("lstm"):
        output, state = lstm(input, state)
    tf.get_variable_scope().reuse_variables()

logit = tf.nn.xw_plus_b(output, dec_w, dec_b)
s_logit = tf.nn.softmax(logit)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit)
tf.get_variable_scope().reuse_variables()

loss = tf.reduce_mean(cross_entropy)

# optimizer = tf.train.AdamOptimizer(learning_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
saver = tf.train.Saver(tf.global_variables())

for i in range(len(train_x)):
    sum_id.append(i)

for j in range(len(val_x)):
    val_id.append(j)

numbers = len(train_x) / len(val_x)
val_id = np.tile(val_id, numbers)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epoch = 100

    for e in range(epoch):
        # new epoch init
        step = 0
        start = 0
        end = start + batch_size

        print 'epoch:', e
        np.random.shuffle(sum_id)
        np.random.shuffle(val_id)
        loss_sum_train = 0
        loss_sum_val = 0
        fl_train = open('loss_train.txt','a')
        fl_val = open('loss_val.txt','a')
        fp_train = open('acc_train.txt', 'a')
        fp_val = open('acc_val.txt', 'a')
        correct_train = 0
        correct_val = 0
        while (end <= len(train_x)):
            current_id = sum_id[start:end]
            current_val_id = val_id[start:end]
            current_val_id = current_val_id.tolist()
            batch_x = np.array(train_x[current_id]).reshape((batch_size, video_frames, pose_num))
            batch_y = train_y[current_id]

            batch_val_x = np.array(val_x[current_val_id]).reshape((batch_size, video_frames, pose_num))
            batch_val_y = val_y[current_val_id]

            if e > 0:
                [loss_val, logit_v] = sess.run([loss, s_logit], feed_dict={video: batch_val_x, label: batch_val_y})
                # print 'val_loss:', loss_val
                loss_sum_val += loss_val

                pred_val_index = np.argmax(logit_v, axis=1)
                label_val_index = np.argmax(batch_val_y, axis=1)
                for i in range(len(pred_val_index)):
                    if pred_val_index[i] == label_val_index[i]:
                        correct_val += 1


            [_, loss_train, logit_val] = sess.run([train_op, loss, s_logit], feed_dict={video: batch_x, label: batch_y})

            LOGDIR = 'log/'
            train_writer = tf.summary.FileWriter(LOGDIR)
            train_writer.add_graph(sess.graph)

            # print 'train_loss:', loss_train
            # print 'val_loss:', loss_val
            # print 'label:', batch_y
            # print 'logit_val:', logit_val
            loss_sum_train += loss_train

            pred_train_index = np.argmax(logit_val, axis=1)
            label_train_index = np.argmax(batch_y, axis=1)
            for i in range(len(pred_train_index)):
                if pred_train_index[i] == label_train_index[i]:
                    correct_train += 1


            start += batch_size
            end = start + batch_size
            step += 1
        if e != 0 and e % 10 == 0:
            saver.save(sess, './lstm_model/pose.model')
            # saver.save(sess, './lstm_model/pose.model', global_step=step)
            print '-------------------save model, e %d' % e

            # print "step:", step

        # deno_train = (video_num_train * repeat_times) / batch_size
        deno_train = len(train_x) / batch_size
        loss_mean_train = loss_sum_train / deno_train
        print "loss_mean_train:", loss_mean_train
        write_train = 'epoch:' + str(e) + ' ' + 'mean_loss_train:' + str(loss_mean_train) + '\n'
        fl_train.write(write_train)

        acc_train = float(correct_train) / len(train_x)
        write_train_acc = 'epoch:' + str(e) + ' ' + 'mean_accuracy_train:' + str(acc_train) + '\n'
        fp_train.write(write_train_acc)


        if e > 0:
            # deno_val = (video_num_val * repeat_times) / batch_size
            deno_val = len(train_x) / batch_size
            loss_mean_val = loss_sum_val / deno_val
            print "loss_mean_val:", loss_mean_val
            write_val = 'epoch:' + str(e) + ' ' + 'mean_loss_val:' + str(loss_mean_val) + '\n'
            fl_val.write(write_val)

            acc_val = float(correct_val) / len(train_x)
            write_val_acc = 'epoch:' + str(e) + ' ' + 'mean_accuracy_train:' + str(acc_val) + '\n'
            fp_val.write(write_val_acc)


fl_train.close()
fl_val.close()
fp_train.close()
fp_val.close()

print 'train finish'

