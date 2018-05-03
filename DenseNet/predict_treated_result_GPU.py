#coding=utf-8
import sys,os,json
import cv2
import math
import numpy as np

img_size = 256 #短边长度256，长边等比例缩放
crop_size = 224
width_height_rate = 0 #长高比例,转换阈值.>=1.2.这里长高比例一定>0，所有图片都执行此步骤

caffe_root = '/data/Experiments/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
os.chdir(caffe_root)
#caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

result = []
result0 = []
#test_dir = '/data/zhangming/aichallenger/aichallenger_val_treated/val/' #val image path
test_dir = '/data/zhangming/aichallenger/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922/'
net_file = '/data/zhangming/aichallenger/densenet/deploy_DenseNet_201.prototxt'
caffe_model = '/data/zhangming/aichallenger/densenet/snapshot/all_no_lmdb_desnet_iter_40000.caffemodel'
#mean_file=caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'

net = caffe.Net(net_file,caffe_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 1/0.017) # from [0,1] to [0,255]
#transformer.set_channel_swap('data', (2,1,0)) #将RGB变换到BGR

test_list = os.listdir(test_dir)
print('----------- %s number is: %d -------------' % (test_dir.split('/')[-2], len(test_list)))
num = 0
for img_name in test_list:
    img_dir = os.path.join(test_dir, img_name)
    
#    im = caffe.io.load_image(img_dir)
    im = cv2.imread(img_dir)
    width, height = im.shape[1],im.shape[0]
    
    if float(max([width, height])) / min([width, height]) > width_height_rate:
        if (width > height): 
            scale = float(img_size) / float(height)
            im = np.array(cv2.resize(np.array(im), (int(width * scale + 1), img_size))).astype(np.float32)
#            im1 = im[:,0:img_size,:]
#            im2 = im[:,im.shape[1]-img_size-1:-1,:]
            im1 = im[0:crop_size, 0:crop_size, :]
            im2 = im[-crop_size:, 0:crop_size, :]
            im3 = im[-crop_size:, -crop_size:, :]
            im4 = im[0:crop_size, -crop_size:, :]
            im5 = im[int(img_size/2)-int(crop_size/2):int(img_size/2)+int(crop_size/2), int(width*scale/2)-int(crop_size/2):int(width*scale/2)+int(crop_size/2), :]
        else:
            scale = float(img_size) / float(width)
            im = np.array(cv2.resize(np.array(im), (img_size, int(height * scale + 1)))).astype(np.float32)
#            im1 = im[0:img_size,:,:]
#            im2 = im[im.shape[0]-img_size-1:-1,:,:]
            im1 = im[0:crop_size, 0:crop_size, :]
            im2 = im[-crop_size:, 0:crop_size, :]
            im3 = im[-crop_size:, -crop_size:, :]
            im4 = im[0:crop_size, -crop_size:, :]
            im5 = im[int(height*scale/2)-int(crop_size/2):int(height*scale/2)+int(crop_size/2), int(img_size/2)-int(crop_size/2):int(img_size/2)+int(crop_size/2), :]
        # im1
        net.blobs['data'].data[...] = transformer.preprocess('data',im1)
        out1 = net.forward()
        prob_all1 = net.blobs['prob'].data[0].flatten() #为0至所有类的概率
  #      np_prob_all1 = np.array([])
  #      for i in prob_all1:
  #          np_prob_all1 = np.append(np_prob_all1, math.exp(i))
        # im2
        net.blobs['data'].data[...] = transformer.preprocess('data',im2)
        out2 = net.forward()
        prob_all2 = net.blobs['prob'].data[0].flatten() #为0至所有类的概率
  #      np_prob_all2 = np.array([])
  #      for i in prob_all2:
  #          np_prob_all2 = np.append(np_prob_all2, math.exp(i))
        # im3
        net.blobs['data'].data[...] = transformer.preprocess('data',im3)
        out3 = net.forward()
        prob_all3 = net.blobs['prob'].data[0].flatten() #为0至所有类的概率
  #      np_prob_all3 = np.array([])
  #      for i in prob_all3:
  #          np_prob_all3 = np.append(np_prob_all3, math.exp(i))
        # im4
        net.blobs['data'].data[...] = transformer.preprocess('data',im4)
        out4 = net.forward()
        prob_all4 = net.blobs['prob'].data[0].flatten() #为0至所有类的概率
  #      np_prob_all4 = np.array([])
  #      for i in prob_all4:
  #          np_prob_all4 = np.append(np_prob_all4, math.exp(i))
        # im5
        net.blobs['data'].data[...] = transformer.preprocess('data',im5)
        out5 = net.forward()
        prob_all5 = net.blobs['prob'].data[0].flatten() #为0至所有类的概率
  #      np_prob_all5 = np.array([])
  #      for i in prob_all5:
  #          np_prob_all5 = np.append(np_prob_all5, math.exp(i))

        prob_sum0 = 0.2*(prob_all1 + prob_all2 + prob_all3 + prob_all4 + prob_all5)
  #      prob_sum = np_prob_all1 + np_prob_all2 + np_prob_all3 + np_prob_all4 + np_prob_all5 #得到所有类的概率,为一个向量
  #      prob_sum = prob_sum/sum(prob_sum) #把概率归一到0-1之间
        
        top_k0 = prob_sum0.argsort()[-1:-4:-1] #top3,返回最大概率对应的位置(即标签)
  #      top_k = prob_sum.argsort()[-1:-4:-1] #top3,返回最大概率对应的位置(即标签)
        
        temp_dict0 = {}
        temp_dict0['image_id'] = img_name
        temp_dict0['label_id'] = top_k0.tolist()
        result0.append(temp_dict0)
   #     temp_dict = {}
   #     temp_dict['image_id'] = img_name
   #     temp_dict['label_id'] = top_k.tolist()
    #    result.append(temp_dict)
        num += 1
   #     print('exp %d %s is %d,%d,%d' % (num, img_name, top_k[0], top_k[1], top_k[2]))
        print('    %d %s is %d,%d,%d' % (num, img_name, top_k0[0], top_k0[1], top_k0[2]))
    else:
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#with open('/data/zhangming/aichallenger/densenet/treated_test_exp_submit.json', 'w') as f:
#    json.dump(result, f)
#    print('write exp_result json, num is %d' % len(result))
with open('/data/zhangming/aichallenger/densenet/treated_test_submit.json', 'w') as f0:
    json.dump(result0, f0)
    print('write result json, num is %d' % len(result0))
