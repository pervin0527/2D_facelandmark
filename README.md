# 2D Face Landmark Detection

This repository reproduces the 2D face landmark detection model I created while working at a company.

The version 1 model is inspired by MobileNetv2 and uses the Inverted Residual Block, and the version 2 model is inspired by EfficientNet and uses the MBConv Block.

In addition, it assists keypoint estimation of the model by performing an additional task of measuring the Euler angle of the face through the Auxiliary Net.

# Performance

| model | NME     | inference time(ms) |
| ----- | ------- | ------------------ |
| v1    | 0.09347 | 0.085              |
| v2    | 0.07253 | 0.0722             |

# Samples

![sample1](./samples/0004.jpg) ![sample2](./samples/0005.jpg) ![sample3](./samples/0008.jpg) ![sample4](./samples/0009.jpg)
