from flask import Flask, render_template
from flask import Flask, Response
import glob
import os
from flask import request


import pickle
import numpy as np

loaded_model = pickle.load(open('nn.sav', 'rb'))

symp = np.load('symptoms.npy')

# PUBLIC URL: http://61.247.181.126:8000
# PUBLIC IP of My PC

# Main script which does all the communication



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# This is the most important
@app.route('/getDisease/<fnam>')
def getFileName(fnam):
    print('haha')
    print(fnam)
    syms = request.args.get('syms')
    syms_list = syms.split(',')
    print(syms)
    print(syms_list)

    try:
        # if the list is valid
        for i in range(len(syms_list)-1):
            ohe_symp = np.zeros((1,132), dtype=np.float32)
            csymp = syms_list[i]
            print(symp)
            ii = symp.tolist().index(csymp)
            print(ii)
            ohe_symp[0,ii] = 1
        out_pred = loaded_model.predict(ohe_symp)
        return str(out_pred[0])
    except:
        # Failed cases
        return 'Could not detect the disease!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


