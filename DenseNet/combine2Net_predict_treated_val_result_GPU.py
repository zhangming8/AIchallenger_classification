#coding=utf-8
import sys,os,json
import cv2
#import matplotlib.pyplot as plt #http://blog.csdn.net/lights_joy/article/details/45933907
import numpy as np
caffe_root = '/data/Experiments/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
os.chdir(caffe_root)
#caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

#%% 设置路径
save_dir = '/data/zhangming/aichallenger/densenet/combine2Net_7p_test_b_submit.json'
#test_dir = '/data/zhangming/aichallenger/aichallenger_val_treated/val/' # val image path
test_dir = '/data/zhangming/aichallenger/ai_challenger_scene_test_b_20170922/scene_test_b_images_20170922/'
#test_dir = '/data/zhangming/aichallenger/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922/'
#test_dir = '/data/face_recongize/aichallenger/aichallenger_val_treated/val/' #val image path
# densenet
net_file_d = '/data/zhangming/aichallenger/densenet/deploy_DenseNet_201.prototxt'
caffe_model_d = '/data/zhangming/aichallenger/densenet/snapshot/finetune_finetune_all_noscale_nolmdb_desnet_iter_20000.caffemodel'
# resnet
net_file_r = '/data/zhangming/aichallenger/resnet152_places365/deploy_resnet152_places365.prototxt'
caffe_model_r = '/data/zhangming/aichallenger/resnet152_places365/snapshot/all_nolmbd_iter_30000.caffemodel'

#%%
result0 = []
img_size = 256 #短边长度256，长边等比例缩放
crop_size = 224
width_height_rate = 0 #长高比例,转换阈值.>=1.2.这里长高比例一定>0，所有图片都执行此步骤

#%% densenet201
#mean_file=caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
net_d = caffe.Net(net_file_d,caffe_model_d,caffe.TEST)
transformer_d = caffe.io.Transformer({'data': net_d.blobs['data'].data.shape})
transformer_d.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
#transformer.set_raw_scale('data', 0.017) # from [0,1] to [0,255],caffe.io.load_image()读进来的是RGB格式和0~1(float)
#transformer.set_channel_swap('data', (2,1,0)) #将RGB变换到BGR

#%% resnet152
#mean_file=caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
net_r = caffe.Net(net_file_r,caffe_model_r,caffe.TEST)
transformer_r = caffe.io.Transformer({'data': net_r.blobs['data'].data.shape})
transformer_r.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
#transformer.set_raw_scale('data', 255) # from [0,1] to [0,255]
#transformer.set_channel_swap('data', (2,1,0)) #将RGB变换到BGR

#%%
test_list = os.listdir(test_dir)
print('----------- %s number is: %d -------------' % (test_dir.split('/')[-2], len(test_list)))
num = 0
for img_name in test_list:
    img_dir = os.path.join(test_dir, img_name)
    
#    im = caffe.io.load_image(img_dir)
    im = cv2.imread(img_dir)
#    plt.imshow(im)
    width, height = im.shape[1],im.shape[0]
    
    if float(max([width, height])) / min([width, height]) > width_height_rate:
        if (width > height): #长>高
            scale = float(img_size) / float(height)
            im = np.array(cv2.resize(np.array(im), (int(width * scale + 1), img_size))).astype(np.float32)
#            im1 = im[:,0:img_size,:]
#            im2 = im[:,im.shape[1]-img_size-1:-1,:]
            im1 = im[0:crop_size, 0:crop_size, :] #左上角
            im2 = im[-crop_size:, 0:crop_size, :] #左下角
            im3 = im[-crop_size:, -crop_size:, :] #右下角
            im4 = im[0:crop_size, -crop_size:, :] #右上角
            im5 = im[int(img_size/2)-int(crop_size/2):int(img_size/2)+int(crop_size/2), int(width*scale/2)-int(crop_size/2):int(width*scale/2)+int(crop_size/2), :] #正中间
            im6 = im[0:crop_size, int(width*scale/2)-int(crop_size/2):int(width*scale/2)+int(crop_size/2), :] #中上方
            im7 = im[-crop_size:, int(width*scale/2)-int(crop_size/2):int(width*scale/2)+int(crop_size/2), :] #中下方
#            im8 = im[int(img_size/2)-int(crop_size/2):int(img_size/2)+int(crop_size/2), 0:crop_size, :] #中左方
#            im9 = im[int(img_size/2)-int(crop_size/2):int(img_size/2)+int(crop_size/2), -crop_size:, :] #中右方
        else: #长<高
            scale = float(img_size) / float(width)
            im = np.array(cv2.resize(np.array(im), (img_size, int(height * scale + 1)))).astype(np.float32)
#            im1 = im[0:img_size,:,:]
#            im2 = im[im.shape[0]-img_size-1:-1,:,:]
            im1 = im[0:crop_size, 0:crop_size, :]
            im2 = im[-crop_size:, 0:crop_size, :]
            im3 = im[-crop_size:, -crop_size:, :]
            im4 = im[0:crop_size, -crop_size:, :]
            im5 = im[int(height*scale/2)-int(crop_size/2):int(height*scale/2)+int(crop_size/2), int(img_size/2)-int(crop_size/2):int(img_size/2)+int(crop_size/2), :]
            im6 = im[int(height*scale/2)-int(crop_size/2):int(height*scale/2)+int(crop_size/2), 0:crop_size, :]
            im7 = im[int(height*scale/2)-int(crop_size/2):int(height*scale/2)+int(crop_size/2), -crop_size:, :]
#            im8 = im[0:crop_size, int(img_size/2)-int(crop_size/2):int(img_size/2)+int(crop_size/2), :]
#            im9 = im[-crop_size:, int(img_size/2)-int(crop_size/2):int(img_size/2)+int(crop_size/2), :]
        # im1
        net_r.blobs['data'].data[...] = transformer_r.preprocess('data',im1)
        out1_r = net_r.forward()
        prob_all1_r = net_r.blobs['prob'].data[0].flatten() #为0至所有类的概率
        
        net_d.blobs['data'].data[...] = transformer_d.preprocess('data',im1)
        out1_d = net_d.forward()
        prob_all1_d = net_d.blobs['prob'].data[0].flatten() #为0至所有类的概率

        # im2
        net_r.blobs['data'].data[...] = transformer_r.preprocess('data',im2)
        out2_r = net_r.forward()
        prob_all2_r = net_r.blobs['prob'].data[0].flatten() #为0至所有类的概率
        
        net_d.blobs['data'].data[...] = transformer_d.preprocess('data',im2)
        out2_d = net_d.forward()
        prob_all2_d = net_d.blobs['prob'].data[0].flatten() #为0至所有类的概率

        # im3
        net_r.blobs['data'].data[...] = transformer_r.preprocess('data',im3)
        out3_r = net_r.forward()
        prob_all3_r = net_r.blobs['prob'].data[0].flatten() #为0至所有类的概率
        
        net_d.blobs['data'].data[...] = transformer_d.preprocess('data',im3)
        out3_d = net_d.forward()
        prob_all3_d = net_d.blobs['prob'].data[0].flatten() #为0至所有类的概率

        # im4
        net_r.blobs['data'].data[...] = transformer_r.preprocess('data',im4)
        out4_r = net_r.forward()
        prob_all4_r = net_r.blobs['prob'].data[0].flatten() #为0至所有类的概率
        
        net_d.blobs['data'].data[...] = transformer_d.preprocess('data',im4)
        out4_d = net_d.forward()
        prob_all4_d = net_d.blobs['prob'].data[0].flatten() #为0至所有类的概率

        # im5
        net_r.blobs['data'].data[...] = transformer_r.preprocess('data',im5)
        out5_r = net_r.forward()
        prob_all5_r = net_r.blobs['prob'].data[0].flatten() #为0至所有类的概率
        
        net_d.blobs['data'].data[...] = transformer_d.preprocess('data',im5)
        out5_d = net_d.forward()
        prob_all5_d = net_d.blobs['prob'].data[0].flatten() #为0至所有类的概率
        
        # im6
        net_r.blobs['data'].data[...] = transformer_r.preprocess('data',im6)
        out6_r = net_r.forward()
        prob_all6_r = net_r.blobs['prob'].data[0].flatten() #为0至所有类的概率
        
        net_d.blobs['data'].data[...] = transformer_d.preprocess('data',im6)
        out6_d = net_d.forward()
        prob_all6_d = net_d.blobs['prob'].data[0].flatten() #为0至所有类的概率
        
        # im7
        net_r.blobs['data'].data[...] = transformer_r.preprocess('data',im7)
        out7_r = net_r.forward()
        prob_all7_r = net_r.blobs['prob'].data[0].flatten() #为0至所有类的概率
        
        net_d.blobs['data'].data[...] = transformer_d.preprocess('data',im7)
        out7_d = net_d.forward()
        prob_all7_d = net_d.blobs['prob'].data[0].flatten() #为0至所有类的概率
        
        # prob
        prob_sum_r = 0.143*(prob_all1_r + prob_all2_r + prob_all3_r + prob_all4_r + prob_all5_r + prob_all6_r + prob_all7_r)
        prob_sum_d = 0.143*(prob_all1_d + prob_all2_d + prob_all3_d + prob_all4_d + prob_all5_d + prob_all6_d + prob_all7_d)
        prob_sum0 = prob_sum_r + prob_sum_d #2两个模型综合输出的概率
        top_k0 = prob_sum0.argsort()[-1:-4:-1] #top3,返回最大概率对应的位置(即标签)
        
        temp_dict0 = {}
        temp_dict0['image_id'] = img_name
        temp_dict0['label_id'] = top_k0.tolist() #top_k0为np数组,tolist()将其转为list
        result0.append(temp_dict0)

        num += 1
        print('%d %s is %d,%d,%d' % (num, img_name, top_k0[0], top_k0[1], top_k0[2]))
    else:
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

with open(save_dir, 'w') as f0:
    json.dump(result0, f0)
    print('write combine2Net_7p_result json, num is %d' % len(result0))
