import cv2
from helpers import yolo_predictions





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

def fastness_detection(options):
    return





def run():
    options = {
        "model": "./Models/Model2/weights/best.onnx",
        "image": "./test_images/92.png",
        "backend": cv2.dnn.DNN_BACKEND_OPENCV,
        "target": cv2.dnn.DNN_TARGET_CPU,
    }

    seat_belt_detection(options)
    fastness_detection(options)

