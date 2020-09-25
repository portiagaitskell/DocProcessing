import numpy as np
import cv2
from pdf2image import convert_from_path
import os

'''
Portia Gaitskell
June 10, 2020
Used to detect checkboxes on typed documents - only works if check is fully contained
'''


# Function used to find the inner of two outlined checkboxes
def find_inner_checkbox(checkboxes, approx, thold=2):
    for i in range(len(checkboxes) - 1, -1, -1):
        box = checkboxes[i]
        box_X, box_Y = split_X_Y(box)

        box_avg = (np.average([max(box_X), min(box_X)]), np.average([max(box_Y), min(box_Y)]))
        box_area = (max(box_X) - min(box_X)) * (max(box_Y) - min(box_Y))

        approx_X, approx_Y = split_X_Y(approx.ravel())

        approx_avg = (np.average([max(approx_X), min(approx_X)]), np.average([max(approx_Y), min(approx_Y)]))
        approx_area = (max(approx_X) - min(approx_X)) * (max(approx_Y) - min(approx_Y))

        # If the centers of the two boxes are within the threshold
        # and if the approx_checkbox has the smaller area, rm the larger box
        # Add the smaller box in the main function
        if abs(box_avg[0] - approx_avg[0]) <= thold and abs(box_avg[1] - approx_avg[1]) <= thold:
            if approx_area < box_area:
                del checkboxes[i]
                return True
            else:
                return False
    return True


# Splits the X and Y pixel coordinates from the checkbox
def split_X_Y(box):
    x = [box[i] for i in range(len(box)) if i%2 == 0]
    y = [box[i] for i in range(len(box)) if i%2 == 1]
    return x, y


# Find the minimum dimensions of the
def minimum_box_dimensions(checkboxes):
    min_height = None
    min_width = None
    for box in checkboxes:
        x, y = split_X_Y(box)

        h = max(y) - min(y)
        w = max(x) - min(x)

        if not min_height or h < min_height:
            min_height = h
        if not min_width or w < min_width:
            min_width = w
    return int(min_height), int(min_width)


# file = .pdf file stored is os.chdir()
# Ratio is the ratio of the arc length of contour - changes approximation of polygon
#   - Note: Ratio may need to be altered depending on results
# delta is the difference between the height and width of the boxes - should be less than 5
# side_length_range may vary depending on data set
# set plot = True to view document with highlighted checkboxes and percent_filled displayed
def checkbox_detection_typed(file, ratio=0.015, delta=4, side_length_range=(10,50), plot=True):
    os.chdir('/Users/portia/Documents/urop/test-doc-processing/data/')

    checkboxes = []

    image = convert_from_path(file)
    for im in image:
        im.save('out.png', 'PNG')

    im = cv2.imread('out.png')
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, ratio*cv2.arcLength(cnt, True), True)

        # only detect squares where polygon is approx by 4 points
        if len(approx) == 4:
            X, Y = split_X_Y(approx.ravel())

            # Get width and height (in pixels) of checkbox
            width = max(X) - min(X)
            height = max(Y) - min(Y)

            if abs(height - width) < delta and side_length_range[0] < width < side_length_range[1]:
                if not checkboxes or find_inner_checkbox(checkboxes, approx):
                    checkboxes.append(list(approx.ravel()))
                    cv2.drawContours(im, [approx], 0, (0, 255, 0))

    print('Number of checkboxes found: {num}'.format(num=len(checkboxes)))

    # Create one dictionary per checkbox - contains num in order, coordinates, percent filled
    # Sort in descending order
    checkbox_dicts = []
    for i in range(len(checkboxes)-1, -1, -1):
        new_dic = dict()
        new_dic['number'] = len(checkboxes)-i
        new_dic['coordinates'] = checkboxes[i]
        new_dic['percent_filled'] = None
        checkbox_dicts.append(new_dic)

    # Take the center of each check box and
    # Find the minimum checkbox size
    min_height, min_width = minimum_box_dimensions(checkboxes)

    w = int(min_width / 2)
    h = int(min_height / 2)

    total = 2 * w * 2 * h
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    for box in checkboxes:
        count = 0
        x, y = split_X_Y(box)
        x_range = (min(x), max(x))
        y_range = (min(y), max(y))

        center = (int((x_range[0] + x_range[1]) / 2), int((y_range[0] + y_range[1]) / 2))

        for i in range(center[0] - w, center[0] + w):
            for j in range(center[1] - h, center[1] + h):
                # fill in area if white - used for debugging
                if threshold[j][i] > 200:
                    threshold[j][i] = 100
                # if pixel is black, add to count
                if threshold[j][i] < 5:
                    count += 1
        percent = round(count / total, 1)

        for dic in checkbox_dicts:
            if dic['coordinates'] == box:
                dic['percent_filled'] = percent

        txt = str(percent)
        cv2.putText(im, txt, (center[0] - 60, center[1] + 5), font, 1.2, (0, 0, 255), thickness=2)
    print(checkbox_dicts)

    if plot:
        dims = im.shape
        print(dims)
        im1 = cv2.resize(im, (int(dims[1]/2.2), int(dims[0]/2.2)))
        im2 = cv2.resize(threshold, (int(dims[1]/2.2), int(dims[0]/2.2)))
        #cv2.imshow('im', im)
        cv2.imshow('im', im1)
        cv2.imshow('threshold', im2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # make a dictionary of checkboxes and percent filled
    return checkbox_dicts


if __name__ == '__main__':
    file1 = 'NY -Resale -Valid -Typed.pdf'
    file2 = 'MTC - Valid -Typed.pdf'

    # Set plot to True to see output
    boxes = (checkbox_detection_typed(file2, plot=True))
    print(boxes)










