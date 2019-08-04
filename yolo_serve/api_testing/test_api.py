import requests
import jsonschema._format
import json
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import core.utils as utils
from PIL import Image
import cv2
import random
import colorsys
import numpy as np

IMAGE_PATH = 'cat_dog.jpg'

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r', encoding = 'utf-8') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

original_image = cv2.imread(IMAGE_PATH)
original_image = cv2.copyMakeBorder(
                 original_image, 
                 20, 
                 20, 
                 20, 
                 20, 
                 cv2.BORDER_CONSTANT,
                 value = [255, 255, 255]
              )
cv2.imwrite('U'+IMAGE_PATH, original_image)
with open('U'+IMAGE_PATH, 'rb') as f:
    im = f.read()

image_string = base64.b64encode(im)
print(image_string)

image_data = image_string #bytes(image_string, encoding="ascii")
#im = Image.open(BytesIO(base64.b64decode(image_data)))
#im.save('test_out.jpg')

r = requests.post('http://A61EX.A247.X181.126:5000/upload', {'image64': image_data}) # encrypt public URL
print(r.status_code)
print(r.text)

rjson = r.json()

rj = json.loads(rjson)

print(type(rj["data"]))

#import io, json
#with io.open('data.json', 'w', encoding='utf-8') as f:
#  f.write(json.dumps(rj, ensure_ascii=False))
bboxes = rj["data"]
print(bboxes)

classes=read_class_names('coco.names')

print('Detected Objects: ')

for obj in bboxes:
    cid = int(obj[5])
    print(classes[cid])


image = utils.draw_bbox(original_image, bboxes)
image = Image.fromarray(image)
image.show()


