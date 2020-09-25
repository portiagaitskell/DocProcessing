import numpy as np
import cv2
import os


#im = cv2.imread('Checkbox.png')
im = cv2.imread('full-doc-checkbox2.png')
imgray = cv2.imread('full-doc-checkbox2.png', cv2.IMREAD_GRAYSCALE)
#imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

count = 0

seen_average = set()

checkboxes = []

thold = 3
def check_seen_average(seen, avg, threshold):
    for elt in seen:
        if abs(avg[0]-elt[0]) <= threshold and abs(avg[1]-elt[1]) <= threshold:
            return False
    return True

# if averages are close remove the one with the larger area
def check_seen_checkboxes(checkboxes, approx, threshold):
    for i in range(len(checkboxes)-1,-1,-1):
        box = checkboxes[i]
        box_X = [box[0], box[2], box[4], box[6]]
        box_Y = [box[1], box[3], box[5], box[7]]

        box_avg = (np.average([max(box_X), min(box_X)]), np.average([max(box_Y), min(box_Y)]))
        box_area = (max(box_X)-min(box_X))*(max(box_Y)-min(box_Y))

        approx_X = [approx.ravel()[0], approx.ravel()[2], approx.ravel()[4], approx.ravel()[6]]
        approx_Y = [approx.ravel()[1], approx.ravel()[3], approx.ravel()[5], approx.ravel()[7]]

        approx_avg = (np.average([max(approx_X), min(approx_X)]), np.average([max(approx_Y), min(approx_Y)]))
        approx_area = (max(approx_X)-min(approx_X))*(max(approx_Y)-min(approx_Y))

        if abs(box_avg[0] - approx_avg[0]) <= thold and abs(box_avg[1] - approx_avg[1]) <= thold:
            print('XX')
            if approx_area < box_area:
                print('II')
                del checkboxes[i]
                return True
            else:
                return False
    return True


for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        X = [approx.ravel()[0], approx.ravel()[2], approx.ravel()[4], approx.ravel()[6]]
        Y = [approx.ravel()[1], approx.ravel()[3], approx.ravel()[5], approx.ravel()[7]]
        width = max(X) - min(X)
        height = max(Y) - min(Y)

        if abs(height - width) < 4 and width > 10:
            cv2.drawContours(im, [approx], 0, (0, 255, 0))


cv2.imshow('image', im)
cv2.imshow('imgray', imgray)
cv2.imshow('threshold', threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()