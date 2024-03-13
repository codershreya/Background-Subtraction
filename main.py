import cv2 #CpenCV
import numpy as np

background_subtractor = cv2.createBackgroundSubtractorKNN()

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not capture.isOpened():
    print("Unable to open camera")
    exit(0)
    
while True:
    _, frame = capture.read()
    if frame is None:
        break
    
    frame = cv2.resize(frame, (600, 600))
    foreground_mask = background_subtractor.apply(frame)
    kernel = np.ones((5,5), np.uint8)
    foreground_mask = cv2.erode(foreground_mask, kernel, iterations=2)
    foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=2)
    
    foreground_mask[np.abs(foreground_mask) < 250] = 0
    
    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground Mask', foreground_mask)
    
    cv2.waitKey(30)