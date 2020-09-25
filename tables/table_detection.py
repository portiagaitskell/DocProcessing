import pytesseract
import cv2
from pdf2image import convert_from_path
import os
import numpy as np
import pandas as pd
from pytesseract import Output


def check_duplicate_boxes(box):
    final_box = []
    #seen = set()
    for (x,y,w,h) in box:
        duplicate = False
        for (a,b,c,d) in final_box:
            if abs(x-a) < 10 and abs(y-b) < 10 and abs(w-c) < 10 and abs(h-d) < 10:
                duplicate = True
        if not duplicate:
            final_box.append([x,y,w,h])

    return final_box


# Two things must be true: find either a box with same x that is within the height
#
def check_table(box):
    tables = []
    # take each box and compare it to all others
    for i, (x,y,w,h) in enumerate(box):
        found = False
        #print(x,y,w,h)
        #print(box[:i]+box[i+1:])
        for (a,b,c,d) in box[:i]+box[i+1:]:
            if abs(x-a) < 10 and not found:
                if min(y, b) == y:
                    height = h
                else:
                    height = d

                if abs(y - b) < height + 5:
                    tables.append([x, y, w, h])
                    found = True

            elif abs(y-b) < 10 and not found:
                if min(x, a) == x:
                    width = w
                else:
                    width = c

                if abs(x - a) < width + 5:
                    tables.append([x, y, w, h])
                    found = True

    return tables


# No single line of boxes allowed
# Each box must have one directly above/below and directly left/right
def check_table_2(box):
    tables = []
    # take each box and compare it to all others
    for i, (x,y,w,h) in enumerate(box):
        found_vertical = False
        found_horizontal = False

        print(x, y, w, h)
        #print(box[:i] + box[i + 1:])

        for (a, b, c, d) in box[:i] + box[i + 1:]:
            if abs(x-a) < 10 and not found_vertical:
                if min(y, b) == y:
                    height = h
                else:
                    height = d

                if abs(y - b) < height + 5:
                    print('Found vertical: {a}, {b}, {c}, {d}'.format(a=a, b=c, c=c, d=d))
                    found_vertical = True

            elif abs(y-b) < 10 and not found_horizontal:
                if min(x, a) == x:
                    width = w
                else:
                    width = c

                if abs(x - a) < width + 5:
                    print('Found horiz: {a}, {b}, {c}, {d}'.format(a=a, b=c, c=c, d=d))
                    found_horizontal = True
        print()

        if found_horizontal and found_vertical:
            tables.append([x, y, w, h])

    return tables


def pdf_to_image(pdf_path):
    images = convert_from_path(pdf_path)
    paths = []
    for i in range(len(images)):
        images[i].save(str(i) + '.jpg', 'JPEG')
        paths.append(str(i) + '.jpg')
    return paths


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


def full_pdf_detection(path):
    im = cv2.imread(path, 0)

    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    d = pytesseract.image_to_data(img, output_type=Output.DICT)

    return d


OFF_LIMITS = ['!', '®', '™', '?']

fpath = '/Users/portia/Documents/urop/test-doc-processing/data/'
file1 = 'MTC-Invalid-Typed.pdf'
file2 = 'NY -Resale -Valid -Typed.pdf'
file3 = 'MTC-V2.pdf'
file4 = 'no_table_example.pdf'
file5 = 'partial_table.pdf'

pdf_path = fpath+file3

im_paths = pdf_to_image(pdf_path)
print(im_paths)

for path in im_paths:

    full_text = full_pdf_detection(path)

    img = cv2.imread(path, 0)

    unmodified_img = cv2.imread(path, 0)

    unmod_img_color = cv2.cvtColor(unmodified_img, cv2.COLOR_BGR2RGB)

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
    #cv2.imwrite("/Users/YOURPATH/horizontal.jpg", horizontal_lines)

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

    print(len(contours))

    for c in contours:
        cv2.drawContours(img, [c], 0, (0, 255, 0))

    dims = img.shape
    im1 = cv2.resize(img, (int(dims[1] / 2.2), int(dims[0] / 2.2)))
    cv2.imshow('im1', im1)

    cv2.waitKey(5000)
    cv2.destroyAllWindows()

    # Create list box to store all boxes in
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (50 < w < 1000 and 25 < h < 500 and w/h > 1.5):
            image = cv2.rectangle(im_color, (x, y), (x + w, y + h), (255, 0, 0), 2)
            box.append([x, y, w, h])

    # Creating two lists to define row and column in which cell is located
    row = []
    column = []
    j = 0
    print(len(box))

    if len(box) == 0:
        print('NO TABLE FOUND')
    else:
        # Sorting the boxes to their respective row and column

        box = check_duplicate_boxes(box)

        #print(box)

        box = check_table_2(box)

        print(box)

        for (x,y,w,h) in box:
            image = cv2.rectangle(im_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

        dims = im_color.shape
        im1 = cv2.resize(im_color, (int(dims[1] / 3), int(dims[0] / 3)))
        cv2.imshow('im1', im1)

        cv2.waitKey(2000)
        cv2.destroyAllWindows()

        print('LENGTH OF TABLES: ' + str(len(box)))

        if len(box) == 0:
            print('NO TABLE FOUND')

        else:

            dims = im_color.shape
            im1 = cv2.resize(im_color, (int(dims[1] / 2.2), int(dims[0] / 2.2)))
            cv2.imshow('im1', im1)

            cv2.waitKey(10)
            cv2.destroyAllWindows()

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
            print(column)
            print(row)

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

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

            # from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
            pd.set_option('display.max_columns', None)
            outer = []
            print(finalboxes)

            cv2.imshow('unmod image', unmodified_img)

            cv2.waitKey(1000)
            cv2.destroyAllWindows()


            for i in range(len(finalboxes)):
                for j in range(len(finalboxes[i])):
                    inner =''
                    if (len(finalboxes[i][j]) == 0):
                        outer.append(' ')
                    else:
                        for k in range(len(finalboxes[i][j])):
                            possible_words = {'a': '', 'b': '', 'c': '', 'd': '', 'final': ''}
                            print(finalboxes[i][j][k])
                            #y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], finalboxes[i][j][k][3]
                            x, y, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                                         finalboxes[i][j][k][3]

                            print(x,y,w,h)

                            full_text_word = ''

                            n_boxes = len(full_text['level'])
                            for n in range(n_boxes):
                                (a, b, c, d) = (full_text['left'][n], full_text['top'][n], full_text['width'][n], full_text['height'][n])
                                if a > x-7 and a < (x+w) and b > y-7 and b < (y+h)-10 and (a+c) < (x+w):
                                    full_text_word = full_text['text'][n]
                                    print('Full text: {t}'.format(t=full_text['text'][n]))

                            img2 = unmod_img_color[y:y + h, x:x + w]
                            print('Img Color: ' + str(pytesseract.image_to_string(img2)))
                            possible_words['a'] = pytesseract.image_to_string(img2)

                            cv2.imshow('img2', img2)
                            cv2.waitKey(1000)
                            cv2.destroyAllWindows()

                            finalimg = unmodified_img[y:y + h, x:x + w]
                            cv2.imshow('im', finalimg)
                            cv2.waitKey(1000)
                            cv2.destroyAllWindows()
                            print('Unmod Img: ' + str(pytesseract.image_to_string(finalimg)))
                            possible_words['b'] = pytesseract.image_to_string(finalimg)

                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                            border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                            resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                            print('Resize: ' + str(pytesseract.image_to_string(resizing)))
                            possible_words['c'] = pytesseract.image_to_string(resizing)

                            dilation = cv2.dilate(resizing, kernel, iterations=1)
                            print('Dilate: ' + str(pytesseract.image_to_string(dilation)))
                            possible_words['d'] = pytesseract.image_to_string(dilation)

                            erosion = cv2.erode(dilation, kernel, iterations=1)

                            out = pytesseract.image_to_string(erosion)
                            print('Erosion: ' + str(out))
                            possible_words['final'] = out

                            #cv2.imshow('im5', erosion)
                            #cv2.waitKey(1000)
                            #cv2.destroyAllWindows()
                            print()

                            if (len(out) == 0):
                                if possible_words['a']:
                                    out = possible_words['a']
                                else:
                                    out = full_text_word
                            #else:
                                #out = pytesseract.image_to_string(erosion, config='--psm 3')

                            new_out = ''
                            for c in list(out):
                                if c not in OFF_LIMITS:
                                    new_out += c

                            inner = inner + " " + new_out
                        outer.append(inner)
            print(outer)

            # Creating a dataframe of the generated OCR list
            arr = np.array(outer)
            print(arr)
            df = pd.DataFrame(arr.reshape(len(row), countcol))
            print(df)
            df.to_csv(fpath+'test_output_table2.csv')

            dims = img_bin.shape
            im1 = cv2.resize(img, (int(dims[1] / 3), int(dims[0] / 3)))
            im2 = cv2.resize(img_bin, (int(dims[1] / 2.2), int(dims[0] / 2.2)))
            im3 = cv2.resize(image_2, (int(dims[1] / 2.2), int(dims[0] / 2.2)))
            im4 = cv2.resize(img_vh, (int(dims[1] / 2.2), int(dims[0] / 2.2)))
            im5 = cv2.resize(bitnot, (int(dims[1] / 2.2), int(dims[0] / 2.2)))

            cv2.imshow('im1', im1)
            cv2.imshow('im', im2)
            cv2.imshow('im2', im3)
            cv2.imshow('im3', im4)
            cv2.imshow('im4', im5)

            cv2.waitKey(0)
            cv2.destroyAllWindows()