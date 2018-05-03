# -*- coding: utf-8 -*-
'''
改变图片长宽之比大于width_height_rate的图片，将其短边变为256，长边按比例缩放；
同时prototxt文件中的data输入层加crop_size:224(目的是让随机裁剪增加样本数量)
'''
import os
import cv2
import numpy as np

img_size = 256
num = 0
# train file tra_label_dir里面为每个图片的路径及标签
tra_label_dir = '/data/zhangming/aichallenger/ai_challenger_scene_validation_20170908/dir_test_CA.txt' # train label
# treated train file
treated_tra_dir = '/data/zhangming/aichallenger/aichallenger_val_treated/val/' #treated train dir
treated_train_label_dir = '/data/zhangming/aichallenger/aichallenger_val_treated/treated_val_label.txt' #treated label

train_CA_label = {}
treated_train_label = open(treated_train_label_dir,'w')
for line in open(tra_label_dir).readlines():
    train_CA_label[line.split(' ')[0]] = line.split(' ')[1]

for img_name in open(tra_label_dir):
    img_name_dir = img_name.split(' ')[0]
    img = cv2.imread(img_name.split(' ')[0],cv2.IMREAD_UNCHANGED)
    img_name = img_name.split('/')[-1]
    width, height = img.shape[1],img.shape[0]
    if (width > height): 
        scale = float(img_size) / float(height)
        img = np.array(cv2.resize(np.array(img), (int(width * scale + 1), img_size))).astype(np.float32)
        cv2.imwrite(treated_tra_dir+img_name.split('.')[0]+'.jpg', img) #new 1 img
        treated_train_label.write(treated_tra_dir+img_name.split('.')[0]+'.jpg'+' '+ train_CA_label[img_name_dir]) #label
    else:
        scale = float(img_size) / float(width)
        img = np.array(cv2.resize(np.array(img), (img_size, int(height * scale + 1)))).astype(np.float32)
        cv2.imwrite(treated_tra_dir+img_name.split('.')[0]+'.jpg', img)
        treated_train_label.write(treated_tra_dir+img_name.split('.')[0]+'.jpg'+' '+ train_CA_label[img_name_dir])
    num += 1
    print('solved (percent):', 100*float(num)/len(train_CA_label))
treated_train_label.close()
print('-----end-----')
