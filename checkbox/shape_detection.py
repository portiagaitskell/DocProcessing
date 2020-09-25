import numpy as np
import cv2


#im = cv2.imread('Checkbox.png')
im = cv2.imread('full-doc-checkbox.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
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
def check_seen_checkboxes(checkboxes, approx, thold):
    for i in range(len(checkboxes)-1,-1,-1):
        box = checkboxes[i]

        box_X, box_Y = split_X_Y(box)

        box_avg = (np.average([max(box_X), min(box_X)]), np.average([max(box_Y), min(box_Y)]))
        box_area = (max(box_X)-min(box_X))*(max(box_Y)-min(box_Y))

        approx_X, approx_Y = split_X_Y(approx.ravel())

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

def split_X_Y(box):
    X = [box[i] for i in range(len(box)) if i%2 == 0]
    Y = [box[i] for i in range(len(box)) if i%2 == 1]

    return X,Y


for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
    # compare index 1 and 3 for length
    # compare index 2 and 4 for width
    #print(approx.ravel())


    if len(approx) == 4:
        X = [approx.ravel()[0], approx.ravel()[2], approx.ravel()[4], approx.ravel()[6]]
        Y = [approx.ravel()[1], approx.ravel()[3], approx.ravel()[5], approx.ravel()[7]]
        width = max(X) - min(X)
        height = max(Y) - min(Y)

        #avg = (np.average([max(X), min(X)]), np.average([max(Y), min(Y)]))

        if abs(height-width) < 3 and width > 10:
            #if checkboxes:
                #print(check_seen_checkboxes(checkboxes, approx, 2))
                #print()
            print(len(checkboxes))
            if not checkboxes or check_seen_checkboxes(checkboxes, approx, 2):
                print(len(checkboxes))
                print()
                count += 1
                cv2.drawContours(im, [approx], 0, (0, 255,0))
                checkboxes.append(list(approx.ravel()))

print(count)
print(checkboxes)
print(len(checkboxes))

# Take the center of each check box and
# Find the minimum checkbox size
min_height = None
min_width = None
for box in checkboxes:
    X = [box[0], box[2], box[4], box[6]]
    Y = [box[1], box[3], box[5], box[7]]

    h = max(Y)-min(Y)
    w = max(X)-min(X)

    if not min_height or h < min_height:
        min_height = h
    if not min_width or w < min_width:
        min_width = w

print(min_width, min_height)
print()
thold = 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
for box in checkboxes:
    count = 0
    X = [box[0], box[2], box[4], box[6]]
    Y = [box[1], box[3], box[5], box[7]]
    x_range = (min(X), max(X))
    y_range = (min(Y), max(Y))

    center = (int((x_range[0]+x_range[1])/2),int((y_range[0]+y_range[1])/2))
    w = int(min_width/2)-thold
    h = int(min_height/2)-thold

    total = 2*w * 2*h
    print(center[0]-w, center[0]+w)
    print(center[1]-h, center[1]+h)

    for i in range(center[0]-w, center[0]+w):
        for j in range(center[1]-h, center[1]+h):
            if threshold[j][i] >200:
                threshold[j][i] = 100
            if threshold[j][i] < 5:
                count += 1
    print(count)
    print(total)
    print()
    txt = str(round(count/total,1))
    cv2.putText(im, txt, (center[0]-15, center[1]+5), font, 1, (0,0,255))
    #cv2.putTe




cv2.imshow('image', im)
cv2.imshow('imgray', imgray)
cv2.imshow('threshold', threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()