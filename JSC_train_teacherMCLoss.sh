#!/bin/bash

CUB='/home/deep/JiabaoWang/data/FGVC/CUB_200_2011/train_test'
CAR='/home/deep/JiabaoWang/data/FGVC/Stanford_Cars/train_test'
AIR='/home/deep/JiabaoWang/data/FGVC/FGVC_Aircraft/train_test'
LAV='/home/deep/JiabaoWang/data/FGVC/FGVC_LAV203/classification'

GPUS='3'
ARCH_SR='srnet'
ARCH_CLS='shufflenetv2'
LR_CLS=0.01
BS=36
SIZE_SRC=224
SIZE_DST=56

for DATA in ${CUB} ${CAR} ${AIR} ${LAV}
do
    if [ ${DATA} = ${CUB} ]; then
        DNAME='cub'
    elif [ ${DATA} = ${CAR} ]; then
        DNAME='car'
    elif [ ${DATA} = ${AIR} ]; then
        DNAME='air'
    else
        DNAME='lav'
    fi

    for ARCH_CLS in 'shufflenetv2'
    do
        for ARCH_SR in 'srnet'
        do
            LOGDIR=log/log_train_teacher_srnet
            NAME=${DNAME}_${ARCH_SR}_${ARCH_CLS}_size${SIZE_DST}
            python JSC_train_teacherMCLoss.py  --logdir $LOGDIR \
                            --name $NAME \
                            --arch_sr $ARCH_SR \
                            --arch_cls $ARCH_CLS \
                            --size_src $SIZE_SRC \
                            --size_dst $SIZE_DST \
                            --batchSize $BS \
                            --dataroot $DATA \
                            --lr_cls $LR_CLS \
                            --cuda $GPUS
        done
    done
done
