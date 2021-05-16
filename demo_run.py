import os

from checkbox import checkbox_detect, checkbox_util
import json
from tables import table_analysis, util
import template_extract

# OUTDATED PATHS - update for new github
blank_fpath = '/Users/portia/Documents/urop/test-doc-processing/demo_03_2021/blank_files/'
template_fpath = '/Users/portia/Documents/urop/test-doc-processing/demo_03_2021/templates/'
output_fpath = '/Users/portia/Documents/urop/test-doc-processing/demo_03_2021/output/'
filled_fpath = '/Users/portia/Documents/urop/test-doc-processing/demo_03_2021/filled_files/'
img_fpath = '/Users/portia/Documents/urop/test-doc-processing/demo_03_2021/img/'


def get_file1(template_fname):
    file1 = 'full_table.pdf'

    pdf_path = blank_fpath + file1
    im_paths = util.pdf_to_image(pdf_path)

    #template_fname = "file1_checkbox.json"

    # result = table_analysis.check_table(im_paths[0])
    # if len(result) > 0:
    #     csv_fname = "file1.csv"
    #     df = table_analysis.read_tables(im_paths[0], result[0], result[1], result[2], fpath=template_fpath,
    #                       csv_name=csv_fname, template_name=template_fname)

    checkbox = checkbox_detect.checkbox_detect(im_paths[0], jsonFile=template_fpath+template_fname,
                                                             fileout=img_fpath+"f1_box")


def read_file1(template_fname):

    file1 = "full_table_FILLED.pdf"

    template_extract.extract_template(filled_fpath + file1, file1[:-4], template_fpath + template_fname, fpath_out=output_fpath,
                     file_out=output_fpath + template_fname)


def get_file2(template_fname):
    file2 = 'partial_table_blank.pdf'

    # FILE 2
    pdf_path = blank_fpath + file2
    im_paths = util.pdf_to_image(pdf_path)

    #result = get_horizontal_lines(im_paths[0], template_fpath+template_fname)
    checkbox = checkbox_detect.checkbox_detect(im_paths[0], jsonFile=template_fpath+template_fname,
                                                             fileout=img_fpath+"f2_box")


def read_file2(template_fname):
    file = "partial_table_FILLED.pdf"

    #template_fname = "file2.json"

    template_extract.extract_template(filled_fpath + file, file[:-4], template_fpath + template_fname, fpath_out=output_fpath,
                     file_out=output_fpath + template_fname)



def get_file3(template_fname):
    file3 = 'alaska-table.pdf'

    pdf_path = blank_fpath + file3
    im_paths = util.pdf_to_image(pdf_path)

    vert_lines = checkbox_util.get_vertical_lines(im_paths[0])

    #result = get_horizontal_lines(im_paths[0], template_fpath + template_fname)
    checkbox = checkbox_detect.checkbox_detect.checkbox_detect(im_paths[0], jsonFile=template_fpath + template_fname,
                                                             boundarylines = vert_lines, fileout=img_fpath+"f3_box")

def read_file3(template_fname):
    file = "alaska-table_FILLED.pdf"

    #template_fname = "file3.json"

    template_extract.extract_template(filled_fpath + file, file[:-4], template_fpath + template_fname, fpath_out=output_fpath,
                     file_out=output_fpath + template_fname)



def test_file3_and_file4():
    file4 = 'CCOEPURCHOFMACH&TOOLS.pdf'

    fpath = '/Users/portia/Desktop/MIT/UROPS/Document Processing/Ryan/Certificates-blank'

    pdf_path = os.path.join(fpath, file4)
    im_paths = util.pdf_to_image(pdf_path)

    vert_lines = checkbox_util.get_vertical_lines(im_paths[0])
    checkbox = checkbox_detect.checkbox_detect(im_paths[0], showLabelBound=True, boundarylines = vert_lines)
    print(checkbox)

    test_file3()


def test_table():
    file4 = 'CCOEPURCHOFMACH&TOOLS.pdf'

    fpath = '/Users/portia/Desktop/MIT/UROPS/Document Processing/Ryan/Certificates-blank'

    pdf_path = os.path.join(fpath, file4)
    im_paths = util.pdf_to_image(pdf_path)

    #result = get_horizontal_lines(im_paths[0], 'alaska-table.jpg')
    checkbox = checkbox_detect.checkbox_detect(im_paths[0], showLabelBound=True, boundarylines=vert_lines)
    print(result)




def test_file3_table():
    fpath = '/Users/portia/Documents/urop/test-doc-processing/demo_files/'
    file3 = 'alaska-table.pdf'

    pdf_path = fpath + file3
    im_paths = util.pdf_to_image(pdf_path)

    #result = get_horizontal_lines(im_paths[0], 'alaska-table.jpg')
    checkbox = checkbox_detect.checkbox_detect(im_paths[0], showLabelBound=True, boundarylines=vert_lines)
    print(result)




def test_allfiles():
    fpath = '/Users/portia/Desktop/MIT/UROPS/Document Processing/Ryan/Certificates-blank'
    for filename in os.listdir(fpath):
        if filename.endswith(".pdf"):
            # print(os.path.join(fpath, filename))
            pdf_path = os.path.join(fpath, filename)
            im_paths = util.pdf_to_image(pdf_path)
            checkbox = checkbox_detect.checkbox_detect(im_paths[0])
            # continue
        else:
            continue

def get_template(fpath, table_fpath, csv_fname, template_fname):
    im_paths = util.pdf_to_image(fpath)

    result = table_analysis.check_table(im_paths[0]) #check for uniform table

    if len(result) > 0:
        table_analysis.read_tables(im_paths[0], result[0], result[1], result[2], fpath=table_fpath,
                                   csv_name=csv_fname, template_name=template_fname)


def final_demo():

    #get_file1()
    #read_file1()

    #get_file2()
    read_file2()
    #
    #get_file3()
    #read_file3()


if __name__ == '__main__':
    read_file3("file3_checkbox.json")