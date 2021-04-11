import numpy as np
import cv2
import pytesseract
from pytesseract import Output
import json


'''
Portia Gaitskell
Mar 5, 2021
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


#updating range to 16-51
# delta from 4 to 12
def checkbox_detect(path, ratio=0.015, delta=12, side_length_range=(16,51), plot=True, fileout=None,
                    jsonFile = None,  showLabelBound=None, boundarylines=None):
    im = cv2.imread(path)
    imgray = cv2.imread(path, 0)

    #imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    checkboxes = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, ratio * cv2.arcLength(cnt, True), True)

        # only detect squares where polygon is approx by 4 points
        if len(approx) == 4:
            #print(approx)
            X, Y = split_X_Y(approx.ravel())

            # Get width and height (in pixels) of checkbox
            top_left = (min(X), min(Y))
            width = max(X) - min(X)
            height = max(Y) - min(Y)

            if abs(height - width) < delta and side_length_range[0] < width < side_length_range[1] and \
                    side_length_range[0] < height < side_length_range[1]:
                if not checkboxes or find_inner_checkbox(checkboxes, approx):
                    checkboxes.append(list(approx.ravel()))
                    #checkboxes.append([])
                    cv2.drawContours(im, [approx], 0, (0, 255, 0))

    print('Number of checkboxes found: {num}'.format(num=len(checkboxes)))

    if len(checkboxes) == 0:
        return

    # Create one dictionary per checkbox - contains num in order, coordinates, percent filled
    # Sort in descending order
    checkbox_dicts = []

    for i in range(len(checkboxes) - 1, -1, -1):
        new_dic = dict()
        new_dic['number'] = len(checkboxes) - i
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

        width = max(x) - min(x)
        height = max(y) - min(y)

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
        #txt2 = str(width) + ", " + str(height)
        cv2.putText(im, txt, (center[0] - 60, center[1] + 5), font, 1.2, (0, 0, 255), thickness=2)
        #cv2.putText(im, txt2, (center[0] - 80, center[1] + 5), font, 1.2, (0, 0, 255), thickness=2)
    #print(checkbox_dicts)

    if fileout:
        name1 = str(fileout) + '_1.jpg'
        cv2.imwrite(name1, im)

    if plot:
        print('Plotting')
        dims = im.shape
        print(dims)
        im1 = cv2.resize(im, (int(dims[1] / 3), int(dims[0] / 3)))
        im2 = cv2.resize(threshold, (int(dims[1] / 2.2), int(dims[0] / 2.2)))
        # cv2.imshow('im', im)
        cv2.imshow('im', im1)

        #cv2.imshow('threshold', im2)

        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    checkbox_dicts = check_overlap(checkbox_dicts)
    clusters = cluster_checkbox(checkbox_dicts, im, showLabelBound, boundarylines)

    print(checkbox_dicts)
    print(clusters)
    get_checkbox_label_basic(path, checkbox_dicts, clusters, fileout=fileout)

    for dic in checkbox_dicts:
        dic['coordinates'] = [int(coord) for coord in dic['coordinates']]

    if jsonFile:
        try:
            with open(jsonFile, "r") as file:
                data = json.load(file)
        except:
            data = {}

        data['checkbox'] = checkbox_dicts

        with open(jsonFile, "w") as file:
            json.dump(data, file)

    # make a dictionary of checkboxes and percent filled
    return checkbox_dicts


def get_checkbox_label_basic(path, checkbox_dicts, clusters, fileout=None):
    im = cv2.imread(path)

    for k, cluster in clusters.items():
        y_upperbound = None
        for i, box in enumerate(cluster["checkboxes"]):

            coords = box['coordinates']

            x, y = split_X_Y(coords)
            x_range = (min(x), max(x))
            y_range = (min(y), max(y))

            try:
                y_lowerbound = y_range[0] + cluster["y gaps"][i]

            except:
                y_lowerbound = y_range[1] + 15

            if y_upperbound is None:
                y_upperbound = y_range[0] - 10

            cv2.rectangle(im, (x_range[1]+11, y_upperbound), (cluster["xlabel_boundary"], y_lowerbound), (0, 255, 0), 2)

            crop = im[y_upperbound: y_lowerbound, x_range[1]+11: cluster["xlabel_boundary"]]

            y_upperbound = y_lowerbound - 15

            if crop.shape[1] != 0:
                cv2.imshow("crop", crop)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()

                label = pytesseract.image_to_string(crop)
                checkbox_dicts[box["number"]]["label"] = label

                d = pytesseract.image_to_data(crop, output_type=Output.DICT)

                label2 = (' ').join([d['text'][i] for i in range(len(d['text']))
                                 if (d['text'][i]!=' ' and d['text'][i] != '' and d['conf'][i] > 75)])

                if len(label2) == 0:
                    label2 = "Error"

                print(label2)
                box['label'] = label2

    if fileout:
        cv2.imshow("final", im)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

        cv2.imwrite(fileout+"_labels.jpg", im)


# reads from output of above file
def checkbox_read(path, checkbox_dict):

    imgray = cv2.imread(path, 0)
    _, threshold = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)

    data = {}

    checkboxes = []
    for dic in checkbox_dict:
        checkboxes.append(dic['coordinates'])

    for box in checkbox_dict:
        count = 0
        #print(box)
        x, y = split_X_Y(box['coordinates'])
        x_range = (min(x), max(x))
        y_range = (min(y), max(y))

        min_height, min_width = minimum_box_dimensions(checkboxes)

        w = int(min_width / 2)
        h = int(min_height / 2)

        total = 2 * w * 2 * h

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

        if percent > 0.15:
            #print(percent)
            data[box['label']] = True

    return data


def check_overlap(checkbox_dicts):
    checkbox_list = []
    remap_dicts = {}
    new_checkbox_dicts = []
    for box in checkbox_dicts:
        coords = box['coordinates']
        checkbox_list.append(coords)

        remap_dicts[tuple(coords)] = box

    final_boxes = []

    overlap_boxes = []

    for i in range(len(checkbox_list)):
        box_i = checkbox_list[i]
        x1, y1 = split_X_Y(box_i)

        for j in range(len(checkbox_list)):
            if i != j:
                box_j = checkbox_list[j]
                x2, y2 = split_X_Y(box_j)

                if not (min(x1) > max(x2) or min(x2) > max(x1)) and not (min(y1) > max(y2) or min(y2) > max(y1)):
                    overlap_boxes.append(box_j)

            #if x1_start >= x2_end or x2_start >= x1_end

        if box_i not in overlap_boxes:
            final_boxes.append(box_i)

    #for each one, if value is

    i = 0
    for box in final_boxes:
        dic = remap_dicts[tuple(box)]
        dic["number"] = i
        i += 1

        new_checkbox_dicts.append(dic)

    return new_checkbox_dicts


# cluster based on x coord
def cluster_checkbox(checkbox_dicts, im=None, showLabelBound=None, boundarylines=None):
    threshold = 3*40
    # for each box, check if
    seen_x = {}
    for box in checkbox_dicts:
        coords = box['coordinates']

        x, y = split_X_Y(coords)
        x_range = (min(x), max(x))


        # check x coords
        if x_range[0] in seen_x:
            seen_x[x_range[0]].append(box)
        else:
            seen_x[x_range[0]] = [box]


    clusters = {}
    count = 0
    # go through seen_y and sort each coordinates[1]
    for k,v in seen_x.items():
        seen_x[k] = sorted(v, key=lambda i: i['coordinates'][1])

        key = "cluster_"+str(count)
        clusters[key] = {}
        clusters[key]['checkboxes'] = []
        clusters[key]['dims'] = [] #stores top left and bottom right, use cv2.rectangle(img, topleft, bottomright, color, thickness)
        clusters[key]['y gaps'] = [] #stores the vertical height between adjacent boxes
        previous_y = None
        for box in seen_x[k]:
            x, y = split_X_Y(box['coordinates'])

            if previous_y is None:
                clusters[key]["checkboxes"].append(box)
                clusters[key]['dims'].append((min(x)-5, min(y)-5))
                previous_y = box['coordinates'][1]
            else:
                #part of cluster
                if abs(box['coordinates'][1] - previous_y) < threshold:
                    clusters[key]["checkboxes"].append(box)
                    clusters[key]['y gaps'].append(abs(box['coordinates'][1] - previous_y))
                    previous_y = box['coordinates'][1]

                else: # create new cluster
                    prev_x, prev_y = split_X_Y(clusters[key]["checkboxes"][-1]["coordinates"])
                    clusters[key]['dims'].append((max(prev_x)+10, max(prev_y)+10))
                    count += 1
                    key = "cluster_" + str(count)
                    clusters[key] = {}
                    clusters[key]["checkboxes"] = [box]
                    clusters[key]['dims'] = [(min(x), min(y))]
                    clusters[key]['y gaps'] = []
                    previous_y = box['coordinates'][1]

        if len(clusters[key]['dims']) == 1:
            prev_x, prev_y = split_X_Y(clusters[key]["checkboxes"][-1]["coordinates"])
            clusters[key]['dims'].append((max(prev_x) + 10, max(prev_y) + 10))

        count += 1

    # create label bondaries
    # for each cluster, seen if the y coords overlap, if they do mark the left side as a boundary for the label
    for k, cluster in clusters.items():
        for k2, cluster2 in clusters.items():
            if k == k2:
                continue
            else:
                #if the x coordinates > current x coord
                if cluster2["dims"][0][0] > cluster["dims"][0][0]:

                    if (cluster["dims"][0][1] <= cluster2["dims"][0][1] <= cluster["dims"][1][1]) or (cluster["dims"][0][1] <= cluster2["dims"][1][1] <= cluster["dims"][1][1]):

                        if "xlabel_boundary" not in cluster:
                            cluster["xlabel_boundary"] = cluster2["dims"][0][0]-20
                        elif cluster["xlabel_boundary"] > cluster2["dims"][0][0]-20:
                            cluster["xlabel_boundary"] = cluster2["dims"][0][0] - 20

                # if the y coordinates overlap, bottom is btw
        if "xlabel_boundary" not in cluster:
            cluster["xlabel_boundary"] = im.shape[1]-20

    if boundarylines is not None:
        for k, cluster in clusters.items():
            # compare to each line x position

            for line in boundarylines:
                p1 = (line[2], line[3])
                p2 = (line[0], line[1])

                # line must be right bound
                if p1[0] > cluster["dims"][0][0]:
                    #check if y values overlap

                    if (p1[1] <= cluster["dims"][0][1]  <= p2[1]) or \
                            (p1[1] <= cluster["dims"][1][1] <= p2[1]):
                        if p1[0] < cluster["xlabel_boundary"]:

                            cluster["xlabel_boundary"] = p1[0]

    #plot clusters
    if im is not None:
        for k, cluster in clusters.items():
            cv2.rectangle(im, cluster["dims"][0], cluster["dims"][1], (255, 0, 0), 3)
            if showLabelBound is not None:
                cv2.rectangle(im, cluster["dims"][0], (cluster["xlabel_boundary"], cluster["dims"][1][1]), (0, 0, 255), 2)

        dims = im.shape
        im1 = cv2.resize(im, (int(dims[1] / 3), int(dims[0] / 3)))
        cv2.imshow('im', im1)

        cv2.waitKey(6000)
        cv2.destroyAllWindows()
    print(count)
    print(clusters)

    return clusters











