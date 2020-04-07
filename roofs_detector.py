from flask import Flask, request, render_template, after_this_request, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import shutil
import os

ROOT_DIR = '%%'  
os.chdir(ROOT_DIR)  # Для импорта библиотек

import datetime
import tensorflow as tf
from libs import roof
import libs.mrcnn.model as modellib
from libs.tools_visualization import get_ax
from libs.mrcnn import visualize
import time

import traceback
from werkzeug.wsgi import ClosingIterator

class AfterResponse:
    def __init__(self, app=None):
        self.callbacks = []
        if app:
            self.init_app(app)

    def __call__(self, callback):
        self.callbacks.append(callback)
        return callback

    def init_app(self, app):
        # install extension
        app.after_response = self

        # install middleware
        app.wsgi_app = AfterResponseMiddleware(app.wsgi_app, self)

    def flush(self):
        for fn in self.callbacks:
            try:
                fn()
            except Exception:
                traceback.print_exc()

class AfterResponseMiddleware:
    def __init__(self, application, after_response_ext):
        self.application = application
        self.after_response_ext = after_response_ext

    def __call__(self, environ, after_response):
        iterator = self.application(environ, after_response)
        try:
            return ClosingIterator(iterator, [self.after_response_ext.flush])
        except Exception:
            traceback.print_exc()
            return iterator


app = Flask("after_response", static_folder='templates/assets')
AfterResponse(app)

@app.after_response
def del_img():

    time.sleep(2)

    try:
        for f in os.listdir(f'{ROOT_DIR}/templates/assets'):
            if f.endswith('.png'):
                os.remove(f'{ROOT_DIR}/templates/assets/{f}')
    except:
        pass

    return None


def detect_roofs(img, img_path):
    with tf.Session() as sess:
        DEVICE = "/cpu:0"
        config = roof.RoofConfig()
        WEIGHTS_PATH = f"{ROOT_DIR}/models/common_model.h5"
        
        class InferenceConfig(config.__class__):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()

        dataset = roof.RoofDataset()
        dataset.add_class("roof", 1, "roof")
        dataset.prepare()

        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=f"{ROOT_DIR}/models/", config=config)
        model.load_weights(WEIGHTS_PATH, by_name=True)

        image = img
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model.detect([image], verbose=1)
        
        ax = get_ax(1)
        r = results[0]

        visualize.save_image(image, 
                            img_path,
                            r['rois'], 
                            r['masks'], 
                            r['class_ids'],
                            r['scores'], 
                            dataset.class_names, 
                            )

        return None


@app.route('/uploads/image_to_detection.png')
def download_file():
    return send_from_directory(f'{ROOT_DIR}/templates/assets/', filename, as_attachment=True)

@app.route('/', methods=['post', 'get'])
def my_form():
    show_img = False
    img_link = ''
    if request.method == 'POST':
        user_file = request.files.getlist('user_file')[0]
        img = user_file.read()
        npimg = np.fromstring(img, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

        img_link = f'image-{str(datetime.datetime.now())}.png'
        img_path = f'{ROOT_DIR}/templates/assets/{img_link}'

        detect_roofs(img, img_path)

        show_img = True

    return render_template('template_roofs.html', show_img=show_img, result_img=img_link)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=99999, debug=True)
