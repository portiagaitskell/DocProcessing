import pytesseract
import cv2
from pdf2image import convert_from_path
import os

def pdf_to_image(pdf_path):
    images = convert_from_path(pdf_path)
    paths = []
    for i in range(len(images)):
        images[i].save(str(i) + '.jpg', 'JPEG')
        paths.append(str(i) + '.jpg')
    return paths

os.chdir('/Users/portia/Documents/urop/test-doc-processing/data/')
fpath = '/Users/portia/Documents/urop/test-doc-processing/data/'
file1 = 'MTC-Invalid-Typed.pdf'

pdf_path = '/Users/portia/Documents/urop/test-doc-processing/data/MTC-Invalid-Typed.pdf'

#print(fpath+file1)
#im_paths = pdf_to_image()
#images = []
#scaled_images = []
#img_cv = cv2.imread(file1)

# By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
# we need to convert from BGR to RGB format/mode:
#img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
#print(pytesseract.image_to_string(img_rgb))

# read pdf using template for given region
im_paths = pdf_to_image(pdf_path)
print(im_paths)

for path in im_paths:
    im = cv2.imread(path, 0)

    img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    print(pytesseract.image_to_string(img_rgb))