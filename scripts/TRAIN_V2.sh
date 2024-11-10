#!/bin/bash
RECORD=2024
WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

#CONFIG=./config/uav-cross-subjectv1/train.yaml
CONFIG=./config/uav-cross-subjectv2/train.yaml

START_EPOCH=50 #50
EPOCH_NUM=70 #60
BATCH_SIZE=56 #56
WARM_UP=5 #5
SEED=666 #777

#python3 main.py --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --only_train_epoch $EPOCH_NUM --seed $SEED
python3 main.py --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 --warm_up_epoch $WARM_UP --only_train_epoch $EPOCH_NUM --seed $SEED
