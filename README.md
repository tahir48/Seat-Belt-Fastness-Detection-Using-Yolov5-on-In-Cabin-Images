# Seat-Belt-Fastness-Detection-Using-Yolov5-on-In-Cabin-Images
This repository contains a seat belt fastness detection algorithm. Yolo-v5 is used for seat belt detection on In-Cabin Images from DriverMVT (Driver Monitoring dataset with Videos and Telemetry) dataset.

## Context
* Data Selection And Annotation
* Yolo Architecture
* Training
* Predictions
* Fastness Detection Rule
* Conclusions

## Data Selection And Annotation
Frames were selected as many different lightning, angle, pose etc. as possible to avoid bias. The final decision making algortihm is chosen to be checking whether centroids of bounded boxes are colinear or not, therefore in annotation phase, whole seat belt is selected into three frames. 
