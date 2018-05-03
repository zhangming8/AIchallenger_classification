# -*- coding: utf-8 -*-


train_path = '/data/zhangming/aichallenger/ai_challenger_scene_train_20170904/scene_train_images_20170904/'
test_path = '/data/zhangming/aichallenger/ai_challenger_scene_validation_20170908/scene_validation_images_20170908/'

# 转换测试集
test = open('/data/zhangming/aichallenger/ai_challenger_scene_validation_20170908/test_CA.txt').readlines()
with open('/data/zhangming/aichallenger/ai_challenger_scene_validation_20170908/dir_test_CA.txt','w') as dir_test:
    for line in test:
        dir_test.write(test_path + line)
        
# 转换训练集        
train = open('/data/zhangming/aichallenger/ai_challenger_scene_train_20170904/train_CA.txt').readlines()
with open('/data/zhangming/aichallenger/ai_challenger_scene_train_20170904/dir_train_CA.txt','w') as dir_train:
    for line in train:
        dir_train.write(train_path + line)
