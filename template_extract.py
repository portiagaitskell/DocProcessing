import pandas as pd
import cv2
from checkbox import checkbox_detect, checkbox_util
import json
from tables import table_analysis, util
import pytesseract


def extract_template(file_in, fname, template_file, fpath_out, file_out):

    im_paths = util.pdf_to_image(file_in)

    path = im_paths[0]

    img = cv2.imread(path, 0)

    output = {}

    with open(template_file, "r+") as file:
        template_json = json.load(file)

    if "uniform_table" in template_json:
        output_df = pd.read_csv(template_json["df_file"], index_col=0)

        for col, dic in template_json["uniform_table"].items():
            for row, finalboxes in dic.items():

                x, y, w, h = finalboxes

                crop = img[y:y + h, x:x + w]

                text = pytesseract.image_to_string(crop)

                #print(row,col)
                #print(output_df.at[int(row), str(col)])

                if len(text) > 0:
                    output_df.at[int(row), str(col)] = text

        #print(output_df)
        df_filename = fname+'.csv'
        output_df.to_csv(fpath_out+df_filename)
        output["df_file"] = fpath_out+df_filename


    if "bounding box" in template_json:
        # contains label, location
        for label, loc in template_json['bounding box'].items():
            x, y, w, h = loc
            crop_image = img[y:y + h, x:x + w]
            # cv2.imshow("crop", crop_image)
            # cv2.waitKey(2000)
            # cv2.destroyAllWindows()
            text = pytesseract.image_to_string(crop_image)

            print(label, text)
            output[label] = text


    if "checkbox" in template_json:
        output["checkbox"] = checkbox_detection_function_2.checkbox_read(path, template_json["checkbox"])


    util.edit_json(file_out, output)

