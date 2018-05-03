1. bash train.sh 得到训练好的模型，在snapshot文件夹，同时记录一下val的准确率
2. 运行 python predict_val_result_GPU.py 得到val_submit.json 文件
3. 运行 python estimate_deploy_right.py 得到对val准确率，并与step1比较，若相等，说明写的deploy文件或调用的预测程序没错
4. 运行 python predict_result_GPU.py 得到最终对test的submit.json标签，即为提交文件
