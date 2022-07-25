# In this script, the functions yolo_predictions, get_detections, non_maximum_supression and 
# and drawings are adopted from https://www.kaggle.com/code/aslanahmedov/automatic-number-plate-recognition
# Since it is Apache License 2.0, and more ethical, I prefer to use it as it is.

import numpy as np
import cv2

def yolo_predictions(img,net):
    input_image, detections = get_detections(img,net)
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    result_img = drawings(img,boxes_np,confidences_np,index)
    return result_img



def get_detections(img,net):
    image = img.copy()
    row, col, d = image.shape
    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image
    blob = cv2.dnn.blobFromImage(input_image,1/255,(640,640),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    return input_image, detections

def non_maximum_supression(input_image,detections):
    ''' Takes input image and detections from get_detections function,
    returns boxes, confidences and indexes of highly candidate bounding boxes
    correspondingly '''

    boxes = []
    confidences = []
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/640
    y_factor = image_h/640
    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5]
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]
                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])
                confidences.append(confidence)
                boxes.append(box)
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)
    return boxes_np, confidences_np, index

def drawings(image,boxes_np,confidences_np,index):
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'Seat_Belt: {:.0f}%'.format(bb_conf*100)
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
    return image


