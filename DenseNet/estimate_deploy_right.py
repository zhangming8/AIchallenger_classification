# -*- coding: utf-8 -*-
# 用来测试自己写的调用训练好的模型进行预测时的程序是否正确
# caffe运行时会有val集的准确率，这里用训练好的模型再次对val集进行预测，如果预测的准确率和
# caffe运行时的准确率一样，说明自己写的预测程序没错
import json

#truth_dir = 'E:/challengerai/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json' #val真实标签
truth_dir = '/data/zhangming/aichallenger/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
#predicted_dir = 'val_submit.json' #用模型预测val生成的标签 准确率为0.9483 第一次提交

#predicted_dir = '../resnet152_places365/treated_val_submit.json' #用模型预测val生成的标签 准确率为0.9507
#predicted_dir = '../resnet152_places365/treated_val_exp_submit.json' #用模型预测val生成的标签,输出的概率取exp() 准确率为0.9488

#predicted_dir = './treated_val_exp_submit.json' 
#predicted_dir = './treated_val_submit.json' 
#predicted_dir = './treated_val_im5_submit.json'

#predicted_dir = './combine2Net_treated_val_submit.json'
#predicted_dir = './treated_val_7p_submit.json'
#predicted_dir = './treated_val_9p_submit.json'
#predicted_dir = './combine2Net_7p_treated_val_submit.json'
predicted_dir = './treated_5p_mirror_val_submit.json'

predicted = {}
truth = {}
num = 0
with open(truth_dir, 'r') as f:
    truth = json.load(f)
with open(predicted_dir, 'r') as f:
    predicted = json.load(f)
    
if len(truth) == len(predicted):
    print('waitting...')
    for pic_t in truth:
        for pic_p in predicted:
            if pic_t['image_id'] == pic_p['image_id']:
                if int(pic_t['label_id']) in pic_p['label_id']:
                    num += 1
    print('the val predicted acc is',float(num)/len(predicted))
else :
    print('xxx Error:the length of truth and predicted are not equal xxx')
    


