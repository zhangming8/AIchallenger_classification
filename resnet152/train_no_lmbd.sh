/data/Experiments/caffe/build/tools/caffe train -solver solver_resnet152_no_lmbd.prototxt --weights=./snapshot/no_lmbd_iter_45000.caffemodel -gpu all 2>&1 | tee logs/all_nolmbd_logs.log
