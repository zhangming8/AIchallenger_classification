#coding=utf-8
import sys,os,json

caffe_root = '/data/Experiments/caffe/'
my_root = '/data/zhangming/aichallenger/'
sys.path.insert(0, caffe_root + 'python')
import caffe
os.chdir(caffe_root)
#caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

result = []
test_dir = my_root + 'ai_challenger_scene_validation_20170908/scene_validation_images_20170908/'
net_file = my_root + 'resnet152_places365/deploy_resnet152_places365.prototxt'                       ##########  
caffe_model = my_root + 'resnet152_places365/snapshot/.caffemodel'                      ##########
#mean_file=caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'

net = caffe.Net(net_file,caffe_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))


test_list = os.listdir(test_dir)
for img_name in test_list:
    im = caffe.io.load_image(os.path.join(test_dir, img_name))
    #im = caffe.io.load_image(caffe_root+'examples/images/cat.jpg')
    net.blobs['data'].data[...] = transformer.preprocess('data',im)
    out = net.forward()
    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-4:-1] #为-4时，得到top3
    temp_dict = {}
    temp_dict['image_id'] = img_name
    temp_dict['label_id'] = top_k.tolist()
    result.append(temp_dict)
    print('image %s is %d,%d,%d' % (img_name, top_k[0], top_k[1], top_k[2]))
with open(my_root + 'resnet152_places365/val_submit.json', 'w') as f:                                       ##########
    json.dump(result, f)
    print('write result json, num is %d' % len(result))
