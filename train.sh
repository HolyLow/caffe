#!/bin/bash

caffe="build/tools/caffe"
stage="train"
solver="models/bvlc_alexnet/mod4_solver.prototxt"
gpu=7
log="mod4_alexnet_test.log"
pretrained="models/bvlc_alexnet/bvlc_alexnet.caffemodel"
$caffe $stage -solver $solver -weights $pretrained -gpu $gpu 2>&1
