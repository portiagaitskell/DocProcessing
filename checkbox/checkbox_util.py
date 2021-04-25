import cv2
import numpy as np
import math


# removes all duplicate boxes
def check_duplicate_lines(lines):
    final_lines = []
    for line in lines:
        duplicate = False
        for final_line in final_lines:
            if abs(line[0]-final_line[0]) < 10 and abs(line[1]-final_line[1])<10 and abs(line[2]-final_line[2]) < 10 and abs(line[3]-final_line[3])<10:
                duplicate = True
        if not duplicate:
            final_lines.append(line)
    return final_lines


# returns all major horizontal lines, used for creating label boundaries
def get_vertical_lines(path, show=False):
    img = cv2.imread(path, 0)

    im_color_line = cv2.imread(path)

    # thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # inverting the image
    img_bin = 255 - img_bin

    # Length(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1] // 50

    # Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

    # Changing to iterations 1 impacts detection of blank lines
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=1)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=1)

    lines = cv2.HoughLinesP(vertical_lines, 30, math.pi / 2, 100, None, 20, 1)

    lines = lines.squeeze()
    lines = check_duplicate_lines(lines)

    for line in lines:
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])

        cv2.line(im_color_line, pt1, pt2, (0, 0, 255), 3)

    dims = im_color_line.shape
    img_resize = cv2.resize(im_color_line, (int(dims[1] / 3), int(dims[0] / 3)))

    if show:
        cv2.imshow('vertical lines', img_resize)
        cv2.waitKey(1000)

    return lines