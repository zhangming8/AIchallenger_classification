# -*- coding: utf-8 -*-
'''
改变图片长宽之比大于width_height_rate的图片，将其短边变为img_size，长边按比例缩放；
缩放的图片分别在长边两位置裁剪，得到两张图片，同时记录其标签
###经过试验测试，效果并不怎样###
'''
import os
import cv2
import numpy as np

img_size = 224
width_height_rate = 1.2 #长高比例,转换阈值.>=1.2的有 51772 张
num = 0
# train file
tra_dir = 'E:/challengerai/ai_challenger_scene_train_20170904/scene_train_images_20170904/' #train dir
tra_label_dir = 'E:/challengerai/ai_challenger_scene_train_20170904/train_CA.txt' # train label
# treated train file
treated_tra_dir = 'E:/challengerai/ai_challenger_scene_train_20170904/ai_challenger_treated_train/' #treated train dir
treated_train_label_dir = 'E:/challengerai/ai_challenger_scene_train_20170904/treated_train_label.txt' #treated label

train_CA_label = {}
treated_train_label = open(treated_train_label_dir,'a')
for line in open(tra_label_dir).readlines():
    train_CA_label[line.split(' ')[0]] = line.split(' ')[1]
    
tra_list = os.listdir(tra_dir)
for img_name in tra_list:
    img = cv2.imread(tra_dir + img_name,cv2.IMREAD_UNCHANGED)
    width, height = img.shape[1],img.shape[0]
    if float(max([width, height])) / min([width, height]) >= width_height_rate: #窄边变为img_size,宽边等比例缩放
        if (width > height): 
            scale = float(img_size) / float(height)
            img = np.array(cv2.resize(np.array(img), (int(width * scale + 1), img_size))).astype(np.float32)
            cv2.imwrite(treated_tra_dir+img_name.split('.')[0]+'_1.jpg',img[:,0:img_size,:]) #new 1 img
            treated_train_label.write(img_name.split('.')[0]+'_1.jpg'+' '+ train_CA_label[img_name]) #label
            cv2.imwrite(treated_tra_dir+img_name.split('.')[0]+'_2.jpg',img[:,img.shape[1]-img_size-1:-1,:]) #new 2 img
            treated_train_label.write(img_name.split('.')[0]+'_2.jpg'+' '+train_CA_label[img_name]) #label
        else:
            scale = float(img_size) / float(width)
            img = np.array(cv2.resize(np.array(img), (img_size, int(height * scale + 1)))).astype(np.float32)
            cv2.imwrite(treated_tra_dir+img_name.split('.')[0]+'_1.jpg',img[0:img_size,:,:])
            treated_train_label.write(img_name.split('.')[0]+'_1.jpg'+' '+ train_CA_label[img_name])
            cv2.imwrite(treated_tra_dir+img_name.split('.')[0]+'_2.jpg',img[img.shape[0]-img_size-1:-1,:,:])
            treated_train_label.write(img_name.split('.')[0]+'_2.jpg'+' '+train_CA_label[img_name])
    else: #原图稍微压缩
        img = np.array(cv2.resize(np.array(img), (img_size, img_size))).astype(np.float32)
        cv2.imwrite(treated_tra_dir+img_name.split('.')[0]+'_0.jpg',img)
        treated_train_label.write(img_name.split('.')[0]+'_0.jpg'+' '+train_CA_label[img_name]) #only size changed
    num += 1
    print('solved (percent):', 100*float(num)/len(train_CA_label))
treated_train_label.close()
    
