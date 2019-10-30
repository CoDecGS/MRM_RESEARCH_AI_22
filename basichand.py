import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while (1):
    ret, frame = cap.read()

    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
    crop_image = frame[100:300, 100:300]

    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lskin = np.array([0, 20, 70])
    uskin = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lskin, uskin)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((5, 5), np.uint8)

    dilation = cv2.dilate(mask, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)





    cv2.imshow("Thresholded", thresh)



    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Erosion', erosion)
    cv2.imshow('Dilation', dilation)

    cv2.imshow('Original', frame)
    edges = cv2.Canny(res, 100, 200)
    cv2.imshow('Edges', edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()