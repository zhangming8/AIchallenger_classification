#coding=utf-8
import sys,os,json
import cv2
import numpy as np

img_size = 256 #短边长度256，长边等比例缩放
crop_size = 224
width_height_rate = 0 #长高比例,转换阈值.>=1.2.这里长高比例一定>0，所有图片都执行此步骤

caffe_root = '/data/Experiments/caffe/'
my_root = '/data/zhangming/aichallenger/'
sys.path.insert(0, caffe_root + 'python')
import caffe
os.chdir(caffe_root)
#caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

result = []
test_dir = my_root + 'aichallenger_val_treated/val/'
#test_dir = '/data/zhangming/aichallenger/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922/'
net_file = my_root + 'resnet152_places365/deploy_resnet152_places365.prototxt'
caffe_model = my_root + 'resnet152_places365/snapshot/all_nolmbd_iter_30000.caffemodel'
#mean_file=caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'

net = caffe.Net(net_file,caffe_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
#transformer.set_raw_scale('data', 255) # from [0,1] to [0,255]
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
        # im2
        net.blobs['data'].data[...] = transformer.preprocess('data',im2)
        out2 = net.forward()
        prob_all2 = net.blobs['prob'].data[0].flatten() #为0至所有类的概率
        # im3
        net.blobs['data'].data[...] = transformer.preprocess('data',im3)
        out3 = net.forward()
        prob_all3 = net.blobs['prob'].data[0].flatten() #为0至所有类的概率
        # im4
        net.blobs['data'].data[...] = transformer.preprocess('data',im4)
        out4 = net.forward()
        prob_all4 = net.blobs['prob'].data[0].flatten() #为0至所有类的概率      
        # im5
        net.blobs['data'].data[...] = transformer.preprocess('data',im5)
        out5 = net.forward()
        prob_all5 = net.blobs['prob'].data[0].flatten() #为0至所有类的概率       
        
        prob_sum = 0.2*(prob_all1 + prob_all2 + prob_all3 + prob_all4 + prob_all5)
        top_k = prob_sum.argsort()[-1:-4:-1] #top3,返回最大概率对应的位置(即标签)
        
        temp_dict = {}
        temp_dict['image_id'] = img_name
        temp_dict['label_id'] = top_k.tolist()
        result.append(temp_dict)
        num += 1
        print('%d %s is %d,%d,%d' % (num, img_name, top_k[0], top_k[1], top_k[2]))
    else:
        im = np.array(cv2.resize(np.array(im), (img_size, img_size))).astype(np.float32)
        net.blobs['data'].data[...] = transformer.preprocess('data',im)
        out = net.forward()
        top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-4:-1] #为-4时，得到top3
        temp_dict = {}
        temp_dict['image_id'] = img_name
        temp_dict['label_id'] = top_k.tolist()
        result.append(temp_dict)
        num += 1
        print('%d image %s is %d,%d,%d' % (num, img_name, top_k[0], top_k[1], top_k[2]))
with open(my_root + 'resnet152_places365/treated_val_submit.json', 'w') as f:
    json.dump(result, f)
    print('write result json, num is %d' % len(result))
