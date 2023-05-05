#!/bin/bash

CUB='/home/deep/JiabaoWang/data/FGVC/CUB_200_2011/train_test'
CAR='/home/deep/JiabaoWang/data/FGVC/Stanford_Cars/train_test'
AIR='/home/deep/JiabaoWang/data/FGVC/FGVC_Aircraft/train_test'

ARCH_SR='srnet'
ARCH_CLS='resnet50'
SIZE_SRC=224
SIZE_DST=56
BS=128
GPUS='3'


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

    for ARCH_CLS in 'resnet18'
    do
        for ARCH_SR in 'srnet'
        do
            LOGDIR=log/log_test_student_srnet
            NAME=${DNAME}_${ARCH_SR}_${ARCH_CLS}_size${SIZE_DST}
            python JSC_test.py  --name $NAME \
                            --arch_sr $ARCH_SR \
                            --arch_cls $ARCH_CLS \
                            --size_src $SIZE_SRC \
                            --size_dst $SIZE_DST \
                            --batchSize $BS \
                            --dataroot $DATA \
                            --cuda $GPUS
        done
    done
done
