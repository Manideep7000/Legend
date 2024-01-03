import os
import numpy as np
from flask import Flask, render_template, request
# import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL.Image import Image

app=Flask(__name__)

model1=load_model("Retinopathy.h5")
@app.route('/')
def index():
    return render_template("index.html")

"""@app.route('/go')
def go():
    return render_template("second.html")"""

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'project',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(120,120))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model1.predict(x),axis=1)
        index=['0','1','2','3','4']
        text="The Retinopathy level is  : " +str(index[pred[0]])
    return render_template('index.html', text1=text)

if __name__=='__main__':
    app.run(debug=True)
