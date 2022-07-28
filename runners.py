import cv2
from helpers import yolo_predictions
from helpers import get_detections
from helpers import non_maximum_supression
from helpers2 import get_centroids
from helpers2 import slope_detector
from helpers2 import colinearity_detector



def seat_belt_detection(options):
    net = cv2.dnn.readNetFromONNX(options['model'])
    net.setPreferableBackend(options['backend'])
    net.setPreferableTarget(options['target'])
    img = cv2.imread(options['image'])
    results = yolo_predictions(img,net)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def fastness_detection(options, methods):
    net = cv2.dnn.readNetFromONNX(options['model'])
    net.setPreferableBackend(options['backend'])
    net.setPreferableTarget(options['target'])
    img = cv2.imread(options['image'])
    input_image, detections = get_detections(img,net)
    boxes_np, confidences_np, index = non_maximum_supression(input_image,detections)
    centroids = get_centroids(img,boxes_np,index)
    if methods["slope"] == True:
        slope_detector(centroids)
    if methods["colinearity"] == True:
        colinearity_detector(centroids)





def run():
    options = {
        "model": "./Models/Model2/weights/best.onnx",
        "image": "./test_images/92.png",
        "backend": cv2.dnn.DNN_BACKEND_OPENCV,
        "target": cv2.dnn.DNN_TARGET_CPU,
    }

    methods = {
        "slope" : True
        "colinearity" : False
    }

    seat_belt_detection(options)
    fastness_detection(options, methods)

