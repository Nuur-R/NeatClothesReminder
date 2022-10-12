import Classifier as cs
import cv2

cap = cv2.VideoCapture(0)
mask = cs.Classifier('MaskModel/keras_model.h5', 'MaskModel/labels.txt')

while True:
    _, img = cap.read()
    predictions, index = mask.getPrediction(img)
    # print(predictions)
    
    cv2.imshow("test", img)
    cv2.waitKey(1)
    
    