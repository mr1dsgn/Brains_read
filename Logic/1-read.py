##########################################################Data loading##########################################################
#Import necessary modules from nipype.
# from nipype.interfaces.slicer import BRAINSFit, BRAINSResample
# import nipype.interfaces.io as nio  # Data i/o
# import nipype.interfaces.utility as util  # utility
# import nipype.pipeline.engine as pe  # pypeline engine
import os  # system functions
# import nibabel as nib
# from nipype.workflows.dmri.fsl.dti import create_eddy_correct_pipeline,\
# create_bedpostx_pipeline
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt



#containers
controls,sample = [],[]
#filling the control set
for subject in os.listdir("../Data/ixi"):
    # print (subject)
    try:
        for dir in os.listdir("../Data/ixi/" + subject):
            # print('-'+dir)
            try:
                for img in os.listdir("../Data/ixi/" + subject+"/" + dir+ "/PD/NIfTI/"):
                    case = nib.load("../Data/ixi/" + subject+"/" + dir+ "/PD/NIfTI/" + img)
                    controls.append(case.get_data())
                    
                    # print(case.shape)
            except:
                pass
    except:
        pass
print("Contols are :", len(controls))

#loading sample data

for subject in os.listdir("../Data/cs_schizbull08"):
    # print (subject)
    try:
        for dir in os.listdir("../Data/cs_schizbull08/" + subject):
            # print('-'+dir)
            try:
                for img in os.listdir("../Data/cs_schizbull08/" + subject+"/" + dir+ "/anat/NIfTI/"):
                    case = nib.load("../Data/cs_schizbull08/" + subject+"/" + dir+ "/anat/NIfTI/" + img)
                    sample.append(case.get_data())
                    # sample = np.append(case.get_data()],sample)
                    # print(case.shape)
            except:
                pass
    except:
        pass
print("samples are :", len(sample))

#find out the range of number of layers
slices = [control.shape[2] for control in controls]
slices.extend(sampl.shape[2] for sampl in sample)
slices = np.array(slices)
mean = np.mean(slices)
median = np.median(slices)
std = np.std(slices)
mini = np.min(slices)
maxi = np.max(slices)
print('max =',maxi)
print('min =',mini)
print('std =',std)
print('median =',median)
print('mean =',mean)

# # Visualising the results
# results = list(rules)
# plt.hist(slices,bins=slices.max())
# plt.show()
completeSet = list(controls)
completeSet.extend(sample)
target = len(controls)*[0]
target.extend(len(sample)*[1])
del controls, sample

sum_of_voxels = []
#voxel information in controls dataset
for case in completeSet[0:578]:
    sum_of_voxels.append(np.sum(case))
sum_of_voxels= np.array(sum_of_voxels)
mean = np.mean(sum_of_voxels)
median = np.median(sum_of_voxels)
std = np.std(sum_of_voxels)
mini = np.min(sum_of_voxels)
maxi = np.max(sum_of_voxels)
del sum_of_voxels
print('max =',maxi)
print('min =',mini)
print('std =',std)
print('median =',median)
print('mean =',mean)
sum_of_voxels = []
#voxel information in sample dataset
for case in completeSet[578:-1]:
    sum_of_voxels.append(np.sum(case))
sum_of_voxels= np.array(sum_of_voxels)
mean = np.mean(sum_of_voxels)
median = np.median(sum_of_voxels)
std = np.std(sum_of_voxels)
mini = np.min(sum_of_voxels)
maxi = np.max(sum_of_voxels)
del sum_of_voxels
print('max =',maxi)
print('min =',mini)
print('std =',std)
print('median =',median)
print('mean =',mean)
#saving as images
X = []
for case in completeSet:
    X.append(np.resize(case,(256,256,158)))
    completeSet.remove(case)
del completeSet
target = np.array(target)
X = np.array(X)
#Saving the Data
np.savez('../Data/DataSet', cases = X, targets = target)








# from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
#                           MaxPooling3D)

# from keras.layers.advanced_activations import LeakyReLU
# from keras.losses import categorical_crossentropy
# from keras.models import Sequential
# from keras.optimizers import Adam
# from keras.utils import np_utils
# from keras.utils.vis_utils import plot_model
# from sklearn.model_selection import train_test_split
# from keras.callbacks import ModelCheckpoint
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session



# def plot_history(history, result_dir):
#     plt.plot(history.history['acc'], marker='.')
#     plt.plot(history.history['val_acc'], marker='.')
#     plt.title('model accuracy')
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.grid()
#     plt.legend(['acc', 'val_acc'], loc='lower right')
#     plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
#     plt.close()

#     plt.plot(history.history['loss'], marker='.')
#     plt.plot(history.history['val_loss'], marker='.')
#     plt.title('model loss')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.grid()
#     plt.legend(['loss', 'val_loss'], loc='upper right')
#     plt.savefig(os.path.join(result_dir, 'model_loss.png'))
#     plt.close()
# def save_history(history, result_dir):
#     loss = history.history['loss']
#     acc = history.history['acc']
#     val_loss = history.history['val_loss']
#     val_acc = history.history['val_acc']
#     nb_epoch = len(acc)

#     with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
#         fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
#         for i in range(nb_epoch):
#             fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
#                 i, loss[i], acc[i], val_loss[i], val_acc[i]))
    

# nb_classes = 2
# ##########################################################pre-processing##########################################################
# model = Sequential()
# model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(256,256,130,1), border_mode='same'))

# model.add(Activation('relu'))
# model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same'))
# model.add(Activation('softmax'))
# model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
# # model.add(Dropout(0.25))
# model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
# model.add(Activation('relu'))
# model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
# model.add(Activation('softmax'))
# model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(512, activation='sigmoid'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes, activation='softmax'))

# model.compile(loss=categorical_crossentropy,
#                 optimizer=Adam(), metrics=['accuracy'])
# model.summary()

#########################################################Train test Split#############################################################
# import os
# import cv2
# import math
# import tensorflow as tf
# import numpy as np

# IMG_SIZE_PX = 50
# SLICE_COUNT = 20
# n_classes = 2
# batch_size = 10
# x = tf.placeholder('float')
# y = tf.placeholder('float')
# keep_rate = 0.8

# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(completeSet, target, test_size=0.33, random_state=42)

# def conv3d(x, W):
#     return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

# def maxpool3d(x):
#     #                        size of window         movement of window as you slide about
#     return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')



# def convolutional_neural_network(x):
#     weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
#                 #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
#                 'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
#                 #                                  64 features
#                 'W_fc':tf.Variable(tf.random_normal([54080,1024])),
#                 'out':tf.Variable(tf.random_normal([1024, n_classes]))}

#     biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
#                 'b_conv2':tf.Variable(tf.random_normal([64])),
#                 'b_fc':tf.Variable(tf.random_normal([1024])),
#                 'out':tf.Variable(tf.random_normal([n_classes]))}
#     x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])
#     conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
#     conv1 = maxpool3d(conv1)


#     conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
#     conv2 = maxpool3d(conv2)

#     fc = tf.reshape(conv2,[-1, 54080])
#     fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
#     fc = tf.nn.dropout(fc, keep_rate)

#     output = tf.matmul(fc, weights['out'])+biases['out']
#     return output

# def train_neural_network(x):
#     prediction = convolutional_neural_network(x)
#     cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
#     optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    
#     hm_epochs = 10
#     with tf.Session() as sess:
#         sess.run(tf.initialize_all_variables())
        
#         successful_runs = 0
#         total_runs = 0
        
#         for epoch in range(hm_epochs):
#             epoch_loss = 0
#             for data in X_train:
#                 total_runs += 1
#                 try:
#                     X = data[0]
#                     Y = data[1]
#                     _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
#                     epoch_loss += c
#                     successful_runs += 1
#                 except Exception as e:
#                     # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one 
#                     # input tensor. Not sure why, will have to look into it. Guessing it's
#                     # one of the depths that doesn't come to 20.
#                     # pass
#                     print(str(e))
            
#             print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

#             correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#             accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

#             print('Accuracy:',accuracy.eval({x:[i[0] for i in X_test], y:[i[1] for i in X_test]}))
            
#         print('Done. Finishing accuracy:')
#         print('Accuracy:',accuracy.eval({x:[i[0] for i in X_test], y:[i[1] for i in X_test]}))
        
#         print('fitment percent:',successful_runs/total_runs)
# train_neural_network(X_train)