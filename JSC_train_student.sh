#!/bin/bash

CUB='/home/deep/JiabaoWang/data/FGVC/CUB_200_2011/train_test'
CAR='/home/deep/JiabaoWang/data/FGVC/Stanford_Cars/train_test'
AIR='/home/deep/JiabaoWang/data/FGVC/FGVC_Aircraft/train_test'

GPUS='1'
ARCH_SR='srnets'
ARCH_CLS_T='resnet50'
LR_CLS=0.01
BS=36
SIZE_SRC=224
SIZE_DST=56

for DATA in ${CUB}
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

    for ARCH_CLS_S in 'resnet18'
    do
        for ARCH_SR in 'srnet'
        do
            LOGDIR=log/log_train_student_srnet
            NAME=${DNAME}_${ARCH_SR}_${ARCH_CLS_S}_size${SIZE_DST}
            python JSC_train_student.py  --logdir $LOGDIR \
                            --name $NAME \
                            --arch_sr $ARCH_SR \
                            --arch_cls_t $ARCH_CLS_T \
                            --arch_cls_s $ARCH_CLS_S \
                            --size_src $SIZE_SRC \
                            --size_dst $SIZE_DST \
                            --batchSize $BS \
                            --dataroot $DATA \
                            --lr_cls $LR_CLS \
                            --cuda $GPUS
        done
    done
done
