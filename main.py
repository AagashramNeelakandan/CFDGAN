#Made for the Final Project by Aagashram Neelakandan
from flask import Flask, render_template
import tensorflow as tf
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from werkzeug.utils import secure_filename
from wtforms import SubmitField, SelectField
from keras.models import load_model
import os
from keras.preprocessing.image import img_to_array, save_img, load_img
from numpy import load,expand_dims
import tensorflow as tf
from tensorflow import Graph #, Session
import shutil
import tempfile
#from google.cloud import storage

import json
import subprocess
import gdown

bucket_name = 'cfdgan.appspot.com'
#directory_name = tempfile.mkdtemp()
directory_name = '/tmp/'
Session = tf.compat.v1.Session
# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)

app.config['SECRET_KEY'] = 'IamIronmAn'
#app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'uploads') # you'll need to create a folder named uploads
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, '/tmp') # you'll need to create a folder named uploads
UPLOAD_FOLDER = '/tmp'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)

MODEL_NAMES = ["10p","50000p","100000p","200000p","500000p","1000000p","10u","100000u","200000u","50000u", "500000u","1000000u"]
AMOUNT_OF_MODELS = len(MODEL_NAMES)
MODELS = []
GRAPHS = []
SESSIONS = []
global flag
flag = 0
global file_url
global filename

model_file_curr_name = ""

with open('model_store_keysIDs.json',"r") as f:
    model_keys = json.load(f)


RE_CHOICES = [('10', 'Re10'),('50000', 'Re50000'),('100000', 'Re100000'),('200000', 'Re200000'),('500000', 'Re500000'),('1000000', 'Re1000000')]
PARAMETER_CHOICES = [('Velocity','Velocity'),('Pressure','Pressure')]

class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image only!'), FileRequired('File was empty!')])
    submit = SubmitField('Upload')
    

class ResultForm(FlaskForm):
    Calculated_Parameters_Select = SelectField(label='Variable Parameter', choices=PARAMETER_CHOICES)
    Re_values_Select = SelectField(label='Re Number', choices=RE_CHOICES)
    get_output = SubmitField('Show Result')
    

def clean_slate():
    folder = directory_name
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)

            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def load_image(filename, size=(512,512)):
	pixels = load_img(filename,target_size=size)
	pixels = img_to_array(pixels)
	pixels = (pixels - 127.5) / 127.5
	pixels = expand_dims(pixels, 0)
	return pixels

def load_models():
    global flag
    for i in range(AMOUNT_OF_MODELS):
       
        load_single_model("model_Final_Re" + MODEL_NAMES[i] + ".h5")


    flag = 1


def load_single_model(path):
    graph = Graph()
    with graph.as_default():
        session = Session()
        with session.as_default():
            model = load_model(path)
            model._make_predict_function() 
            
            MODELS.append(model)
            GRAPHS.append(graph)
            SESSIONS.append(session)


def download_model(model_name):
    
    output_file="model_Final_Re" + model_name + ".h5"
    try:
        if not os.path.exists(output_file):
            url = "https://drive.google.com/uc?id="+model_keys[model_name]
            gdown.download(url,output_file,quiet=True)
            print("Download completed "+model_name)
    except Exception as e:
        print("Error in download")
        print(e)
        return False

    return True
        
        
    
    
def delete_file():
    try:
        for file in os.listdir():
            if file.endswith(".h5"):
                os.remove(file)
    except:
        return False

    return True

def models_predict(file_path,model_selected_name):
    # Beware to adapt the preprocessing to the input of your trained models
    x = load_image(directory_name+file_path)
    preds = []
    i=0

    with GRAPHS[i].as_default():
        with SESSIONS[i].as_default():
            gen_image = MODELS[i].predict(x)
            save_img(directory_name+file_path.split('.')[0]+model_selected_name+'.png',gen_image[0])
            preds.append(gen_image)

    return preds


@app.route('/', methods=['GET'])
def index():

    form_upload = UploadForm()
    form_result = ResultForm()
    return render_template('index.html',form_upload = form_upload,form_result = form_result)

@app.route('/upload', methods=['GET','POST'])
def upload():
    global flag
    global filename
    global file_url
    os.system("sudo rm -rf /tmp/*")
    clean_slate()
    form_upload = UploadForm()
    form_result = ResultForm()
    if form_upload.validate_on_submit():
        filename = photos.save(form_upload.photo.data)

        file_url = photos.url(filename)
  
      
    else:
        file_url = None
        filename = None
        result_url=None
       
    return render_template('index.html',form_upload = form_upload,form_result = form_result, file_url=file_url,filename = filename)#,file1=file1,file2 = file2)

@app.route('/show_result', methods=['GET','POST'])
def show_result():
    global file_url
    global filename
    global model_file_curr_name
    form_upload = UploadForm()
    form_result = ResultForm()
    file_result = filename
    selected_Re = form_result.Re_values_Select.data
    selected_Parameter = form_result.Calculated_Parameters_Select.data
    if selected_Parameter == 'Pressure':
        para = 'p'
    else:
        para = 'u'

    if(len(MODELS) != 0):
        MODELS.pop()
        GRAPHS.pop()
        SESSIONS.pop()
        
    
    model_selected_name = str(selected_Re)+para
##    load_single_model("models/model_Final_Re" + model_selected_name + ".h5")
            
    model_file_name = "model_Final_Re" + model_selected_name + ".h5"

##    Downloading Model
    if not os.path.exists(model_file_name):
        delete_file()
        download_model(model_selected_name)

    model_file_curr_name = model_file_name
    
    load_single_model(model_file_name)
    predictions = models_predict(filename,model_selected_name)
    result_file = file_result.split('.')[0]+str(selected_Re)+para+'.png'
    result_url = photos.url(result_file)
    
        

    return render_template('index.html',form_upload = form_upload,form_result=form_result, file_url=file_url,result_url=result_url,result_file = result_file,file_result = file_result)


##if __name__ == '__main__':
##    #app.debug=True
##    #app.run()
##    app.run(host='127.0.0.1', port=8080, debug=True)

