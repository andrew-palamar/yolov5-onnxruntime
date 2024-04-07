#!/bin/bash

# Webcam Video
#build/yolo_ort --model_path models/yolov5s.onnx --v4l2 /dev/video0 --class_names models/coco.names

# Single Image
# build/yolo_ort --model_path models/yolov5m_simplified.onnx --image images/frame.bmp --class_names models/coco.names
build/yolo_ort --model_path models/yolov5m.onnx --image images/frame.bmp --class_names models/coco.names --show
