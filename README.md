# 基于超分与知识蒸馏的低分辨率装甲车辆细粒度识别方法

## 环境配置

> Python 3

> Pytorch 1.7.1

> torchvision 0.8.2

## 训练

```sh
./JSC_train_teacher.sh
```
请根据需要修改JSC_train_teacher.sh和JSC_train_teacher.py中的模型路径。

```sh
./JSC_train_student.sh
```
请根据需要修改JSC_train_student.sh和JSC_train_student.py中的模型路径。

## 测试

```sh
./JSC_test.sh
```

## 评测结果

![Comparison Results Table](./data/result.jpg)
