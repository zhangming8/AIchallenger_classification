/data/Experiments/caffe/build/tools/caffe train -solver solver_resnet152.prototxt --weights=resnet152_places365.caffemodel -gpu all 2>&1 | tee logs/my_logs.log
