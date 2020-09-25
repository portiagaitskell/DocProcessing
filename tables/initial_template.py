
import pytesseract
import cv2
from pdf2image import convert_from_path
import os
import numpy as np
import pandas as pd
from pytesseract import Output
from pythonRLSA import rlsa
import math
import copy


def pdf_to_image(pdf_path):
    images = convert_from_path(pdf_path)
    paths = []
    for i in range(len(images)):
        images[i].save(str(i) + '.jpg', 'JPEG')
        paths.append(str(i) + '.jpg')
    return paths


# Checks that there are no duplicate boxes
def check_duplicate_boxes(box):
    final_box = []

    for (x, y, w, h) in box:
        duplicate = False
        for (a, b, c, d) in final_box:
            if abs(x-a) < 10 and abs(y-b) < 10 and abs(w-c) < 10 and abs(h-d) < 10:
                duplicate = True
        if not duplicate:
            final_box.append([x,y,w,h])

    return final_box


# No single line of boxes allowed
# Each box must have one directly above/below and directly left/right
# Returns all boxes in the table
def return_table(box):
    tables = []
    # take each box and compare it to all others
    for i, (x, y, w, h) in enumerate(box):
        found_vertical = False
        found_horizontal = False

        for (a, b, c, d) in box[:i] + box[i + 1:]:
            if abs(x-a) < 10 and not found_vertical:
                if min(y, b) == y:
                    height = h
                else:
                    height = d

                if abs(y - b) < height + 5:
                    found_vertical = True

            elif abs(y-b) < 10 and not found_horizontal:
                if min(x, a) == x:
                    width = w
                else:
                    width = c

                if abs(x - a) < width + 5:
                    found_horizontal = True

        if found_horizontal and found_vertical:
            tables.append([x, y, w, h])

    return tables


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def check_table(path):
    img = cv2.imread(path, 0)

    im_color = cv2.imread(path)

    # thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # inverting the image
    img_bin = 255 - img_bin

    # Length(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1] // 100
    # Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Changing to iterations 1 impacts detection of blank lines
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=1)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=1)

    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=1)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=2)
    # cv2.imwrite("/Users/YOURPATH/horizontal.jpg", horizontal_lines)

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    # Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    _, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)

    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method='top-to-bottom')

    # Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    # Get mean of heights
    mean = np.mean(heights)

    # Create list box to store all boxes in
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (50 < w < 1000 and 25 < h < 500 and w / h > 1.5):
            image = cv2.rectangle(im_color, (x, y), (x + w, y + h), (255, 0, 0), 2)
            box.append([x, y, w, h])

    # Creating two lists to define row and column in which cell is located
    row = []
    column = []

    if len(box) == 0:
        print('NO TABLE FOUND')
        return []
    else:
        # Sorting the boxes to their respective row and column

        box = return_table(check_duplicate_boxes(box))

        if len(box) == 0:
            print('NO TABLE FOUND')
            return []

        else:

            for i in range(len(box)):
                for (x, y, w, h) in box:
                    cv2.rectangle(im_color, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if (i == 0):
                    column.append(box[i])
                    previous = box[i]
                else:
                    if (box[i][1] <= previous[1] + mean / 2):
                        column.append(box[i])
                        previous = box[i]
                        if (i == len(box) - 1):
                            row.append(column)
                    else:
                        row.append(column)
                        column = []
                        previous = box[i]
                        column.append(box[i])
            #print(column)
            #print(row)

            # calculating maximum number of cells
            countcol = 0
            for i in range(len(row)):
                countcol = len(row[i])
                if countcol > countcol:
                    countcol = countcol

            # Retrieving the center of each column
            center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
            center = np.array(center)
            center.sort()

            # Regarding the distance to the columns center, the boxes are arranged in respective order
            finalboxes = []

            for i in range(len(row)):
                lis = []
                for k in range(countcol):
                    lis.append([])
                for j in range(len(row[i])):
                    diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
                    minimum = min(diff)
                    indexing = list(diff).index(minimum)
                    lis[indexing].append(row[i][j])
                finalboxes.append(lis)

            print('Table found')
            finalboxes = np.squeeze(np.array(finalboxes))

            return [finalboxes, row, countcol]


def get_table_coords(boxes):
    cols, rows, num = boxes.shape

    top_left = boxes[0][0]
    top_right = boxes[cols-1][rows-1]

    x = top_left[0]
    y = top_left[1]
    w = top_right[2] + top_right[0] - top_left[0]
    h = top_right[3] + top_right[1] - top_left[1]

    return [x,y,w,h]


fpath = '/Users/portia/Documents/urop/test-doc-processing/data/'
file1 = 'MTC-Invalid-Typed.pdf'
file2 = 'NY -Resale -Valid -Typed.pdf'
file3 = 'MTC-V2.pdf'
file4 = 'no_table_example.pdf'
file5 = 'partial_table.pdf'


#file = file1

pdf_path = fpath+file1

im_paths = pdf_to_image(pdf_path)
print(im_paths)

for path in im_paths:

    image = cv2.imread(path) # reading the image

    tables = check_table(path)

    table_coords = []
    if tables:
        table_coords = get_table_coords(tables[0])

    print(table_coords)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert2grayscale
    (thresh, binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # convert2binary
    cv2.imshow('binary', binary)
    #cv2.imwrite('binary.png', binary)

    (contours, _) = cv2.findContours(~binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # find contours
    '''
    for contour in contours:
        """
        draw a rectangle around those contours on main image
        """
        [x, y, w, h] = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    '''

    cv2.imshow('contour', image)
    #cv2.imwrite('contours.png', image)

    mask = np.ones(image.shape[:2], dtype="uint8") * 255  # create blank image of same dimension of the original image
    (contours, _) = cv2.findContours(~binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(contour)[3] for contour in contours]  # collecting heights of each contour
    avgheight = sum(heights) / len(heights)  # average height

    image_height, image_width = image.shape[:2]
    # finding the larger contours
    # Applying Height heuristic
    # Must be in top third of document

    for c in contours:
        [x, y, w, h] = cv2.boundingRect(c)
        #if h > 1.6 * avgheight and y < image_height / 3:
        if h > 1 * avgheight and y < image_height/7:
            cv2.drawContours(mask, [c], -1, 0, -1)
    cv2.imshow('filter', mask)
    #cv2.imwrite('filter.png', mask)

    # Apply RLSA to combine title into one
    x, y = mask.shape
    value = max(math.ceil(x / 50), math.ceil(y / 50)) + 20  # heuristic
    mask = rlsa.rlsa(mask, True, False, value)  # rlsa application
    cv2.imshow('rlsah', mask)
    cv2.imwrite('rlsah.png', mask)


    # Applying width heuristic
    (contours, _) = cv2.findContours(~mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    #(contours, _) = cv2.findContours(~full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask2 = np.ones(image.shape, dtype="uint8") * 255  # blank 3 layer image
    full_title = ''

    contours, bboxes = sort_contours(contours)

    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)

        use_contour = True

        if len(table_coords) > 0:
            #print(abs(x-table_coords[0]))
            #print(abs(y-table_coords[1]))
            #print(abs(w-table_coords[2]))
            #print()
            if abs(x-table_coords[0]) < 10 and abs(y-table_coords[1]) < 10 and abs(w-table_coords[2]) < 15 \
                    and abs(h-table_coords[3]) < 15:
                use_contour = False

        if w > 0.2 * image.shape[1] and w < 0.8 * image.shape[1] and use_contour:  # width heuristic applied
            print(x, y, w, h)
            title = image[y: y + h, x: x + w]
            cv2.rectangle(mask2, (x, y), (x + w, y + h), (255, 0, 0), 2)

            read_title = image[y-5: y + h+5, x-5: x + w+5]

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            border = cv2.copyMakeBorder(read_title, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
            resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            print('Resize: ' + str(pytesseract.image_to_string(resizing)))

            dilation = cv2.dilate(resizing, kernel, iterations=1)
            print('Dilate: ' + str(pytesseract.image_to_string(dilation)))
            cv2.imshow('dilate', dilation)

            erosion = cv2.erode(dilation, kernel, iterations=1)
            cv2.imshow('erosion', erosion)

            out = pytesseract.image_to_string(erosion)

            full_title += out + ' '

            print(out)


            mask2[y: y + h, x: x + w] = title  # copied title contour onto the blank image
            image[y: y + h, x: x + w] = 255  # nullified the title contour on original image

    print(full_title)

    dims = mask2.shape
    im = cv2.resize(mask2, (int(dims[1] / 2.2), int(dims[0] / 2.2)))
    cv2.imshow('title', im)
    #cv2.imwrite('title.png', mask2)
    cv2.imshow('content', image)
    #cv2.imshow('content.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # mask — ndarray we got after applying rlsah
    # mask2 — blank array

