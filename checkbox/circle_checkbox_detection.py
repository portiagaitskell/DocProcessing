import numpy as np
import cv2
import math
from pdf2image import convert_from_path
import os

'''
Portia Gaitskell
June 17, 2020
Used to detect checkboxes on typed documents - only works if check is fully contained
'''

def circle_coverage(x, y, r):
    indices = []
    for i in range(x-r, x+r):
        for j in range(y-r, y+r):
            dx = x-i
            dy = y-j
            dist = math.sqrt((dx ** 2) + (dy** 2))
            #print(dist)
            if dist <= r:
                indices.append((i,j))
    return indices

# Runs for one page of file passed
# Needs to be modified to handle multiple pages (if required)
# Modify necessary parameters to reduce errors/noise
def circle_checkbox_typed(file, page_num=2, dp=1, minDist=10, param1=50, param2=30, minRadius=20, maxRadius=30, plot=True):

    os.chdir('/Users/portia/Documents/urop/test-doc-processing/data/')

    image = convert_from_path(file)

    for i, im in enumerate(image):
        im.save('out_{i}.png'.format(i=i+1), 'PNG')

    im = cv2.imread('out_{i}.png'.format(i=page_num))
    output = im.copy()
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imgray = cv2.medianBlur(imgray, 5)
    _, threshold = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)

    circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    detected_circles = np.uint16(np.around(circles))

    print('Detected {n} circles'.format(n=len(detected_circles[0, :])))

    checkbox_dicts = []
    for i in range(len(detected_circles[0, :])-1, -1, -1):
        new_dic = dict()
        (x,y,r) = detected_circles[0, :][i]
        new_dic['center'] = (x,y)
        new_dic['radius'] = r
        new_dic['percent_filled'] = None

        checkbox_dicts.append(new_dic)

    radii = []
    for (x,y,r) in detected_circles[0, :]:
        radii.append(r)

    min_r = min(radii)

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    for (x, y, r) in detected_circles[0, :]:

        cv2.circle(output, (x,y), r, (0,255,0), 3)

        count = 0
        total = math.pi*(r**2)
        for p in circle_coverage(x, y, min_r-int(min_r/4)):
            i = p[0]
            j = p[1]
            if threshold[j][i] > 200:
                threshold[j][i] = 100
            # if pixel is black, add to count
            if threshold[j][i] < 5:
                count += 1
                output[j][i] = (255,0,0)

        percent = round(count / total, 3)

        for dic in checkbox_dicts:
            if dic['center'] == (x,y):
                dic['percent_filled'] = percent

        if plot:
            txt = str(percent)
            cv2.putText(output, txt, (x - 150, y + 5), font, 2, (0, 0, 255), thickness=2)

    if plot:
        dims = output.shape
        im1 = cv2.resize(output, (int(dims[1]/4), int(dims[0]/4)))
        im2 = cv2.resize(threshold, (int(dims[1]/4), int(dims[0]/4)))

        cv2.imshow('output',im1)
        cv2.imshow('threshold', im2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return checkbox_dicts


if __name__ == '__main__':
    file = 'CT-Circle.pdf'
    print(circle_checkbox_typed(file, plot=False))