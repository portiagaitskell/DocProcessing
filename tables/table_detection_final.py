import pytesseract
import cv2
from pdf2image import convert_from_path
import os
import numpy as np
import pandas as pd


# Used to extract image from pdf
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

        #for (x, y, w, h) in box:
        #    image = cv2.rectangle(im_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #dims = im_color.shape
        #im1 = cv2.resize(im_color, (int(dims[1] / 3), int(dims[0] / 3)))
        #cv2.imshow('im1', im1)

        #cv2.waitKey(2000)
        #cv2.destroyAllWindows()

        #print('LENGTH OF TABLES: ' + str(len(box)))

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
            return [finalboxes, row, countcol]


def read_tables(path, finalboxes, row, countcol):

    unmodified_img = cv2.imread(path, 0)

    # from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    pd.set_option('display.max_columns', None)
    outer = []

    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner = ''
            if (len(finalboxes[i][j]) == 0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                                 finalboxes[i][j][k][3]

                    finalimg = unmodified_img[x:x + h, y:y + w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                    dilation = cv2.dilate(resizing, kernel, iterations=1)

                    erosion = cv2.erode(dilation, kernel, iterations=1)

                    out = pytesseract.image_to_string(erosion)

                    if (len(out) == 0):
                        out = pytesseract.image_to_string(erosion, config='--psm 3')
                    inner = inner + " " + out
                outer.append(inner)
    #print(outer)

    # Creating a dataframe of the generated OCR list
    arr = np.array(outer)
    #print(arr)
    df = pd.DataFrame(arr.reshape(len(row), countcol))
    return df
    # df.to_csv(fpath+'test_output_table2.csv')


if __name__ == '__main__':
    fpath = '/Users/portia/Documents/urop/test-doc-processing/data/'
    file1 = 'MTC-Invalid-Typed.pdf'
    file2 = 'NY -Resale -Valid -Typed.pdf'
    file3 = 'MTC-V2.pdf'
    file4 = 'no_table_example.pdf'
    file5 = 'partial_table.pdf'

    pdf_path = fpath + file1

    im_paths = pdf_to_image(pdf_path)
    #print(im_paths)

    for path in im_paths:
        result = check_table(path)
        if len(result) > 0:
            print(read_tables(path,  result[0], result[1], result[2]))


