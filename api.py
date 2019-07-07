from flask import Flask
from flask_restful import Resource, Api, reqparse
import werkzeug, os
import json
import numpy as np

from pdf2image import convert_from_path, convert_from_bytes

from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



app = Flask(__name__)
api = Api(app)
UPLOAD_FOLDER = 'pdfs/'
parser = reqparse.RequestParser()
parser.add_argument('file',type=werkzeug.datastructures.FileStorage, location='files')


class HelloWorld(Resource):
    def get(self):
        return {'status': 'ok'}


class PhotoUpload(Resource):
    decorators=[]

    def post(self):
        data = parser.parse_args()
        if data['file'] == "":
            return {
                    'data':'',
                    'message':'No file found',
                    'status':'error'
                    }
        pdffile = data['file'].read()


        if pdffile:
            filename = 'pdf_now.pdf'
            #with open(filename, 'wb') as f:
            #    f.write(pdffile)

            open(filename, 'wb').write(pdffile)

            images = convert_from_path(filename, dpi=20)

            images = [np.array(a) for a in images]



            return json.dumps({
                    'data': (images),
                    'message':'pdf uploaded',
                    'status':'success'
                    }, cls=NumpyEncoder)
        return {
                'data':'',
                'message':'Something when wrong',
                'status':'error'
                }


api.add_resource(HelloWorld, '/test')
api.add_resource(PhotoUpload,'/upload')

if __name__ == '__main__':
    app.run(debug=True)