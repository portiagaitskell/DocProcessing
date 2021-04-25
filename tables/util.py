import json
from pdf2image import convert_from_path

def edit_json(jsonFile, newData):
    try:
        with open(jsonFile, "r+") as file:
            data = json.load(file)
            data.update(newData)
            file.seek(0)
            json.dump(data, file)

    except:
        with open(jsonFile, "w") as file:
            json.dump(newData, file)


def pdf_to_image(pdf_path):
    images = convert_from_path(pdf_path)
    paths = []
    for i in range(len(images)):
        images[i].save(str(i) + '.jpg', 'JPEG')
        paths.append(str(i) + '.jpg')
    return paths