#coding=utf-8
import sys,os

caffe_root = '/data/Experiments/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np

mean_prototxt = '/data/zhangming/aichallenger/creat_lmdb/mean.binaryproto'
mean_npy_dir = '/data/zhangming/aichallenger/creat_lmdb/mean.npy'

blob = caffe.proto.caffe_pb2.BlobProto()           # 创建protobuf blob
data = open(mean_prototxt, 'rb' ).read()       # 读入mean.binaryproto文件内容
blob.ParseFromString(data)                         # 解析文件内容到blob

array = np.array(caffe.io.blobproto_to_array(blob))
mean_npy = array[0]
np.save(mean_npy_dir ,mean_npy)

