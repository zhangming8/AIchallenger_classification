/data/Experiments/caffe/build/tools/caffe train -solver solver_DenseNet_201.prototxt --weights ./snapshot/finetune__all_noscale_nolmdb_desnet_iter_14000.caffemodel -gpu all 2>&1 | tee logs/all_noscale_densenet_logs.log
