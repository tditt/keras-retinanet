import sys
import time

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pyscreenshot
import tensorflow as tf
from system_hotkey import SystemHotkey

from keras_retinanet.models import load_model
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption


def _get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


print('initializing session and model...')
sess = _get_session()
keras.backend.tensorflow_backend.set_session(sess)
stay_looped = True
model = load_model('snapshots/inference_resnet50_csv_50.h5', backbone_name='resnet50')
graph = tf.get_default_graph()
print('...done!')


def main():
    global stay_looped
    hk = SystemHotkey()
    hk.register(('control', 'f8'), callback=lambda x: _capture_and_detect())
    hk.register(('control', 'f12'), callback=lambda y: _stop_loop())
    while stay_looped:
        time.sleep(1)
    sys.exit('exiting...')


def _stop_loop():
    global stay_looped
    stay_looped = False


def _capture_and_detect():
    global model
    global sess
    global graph
    print('capturing screen...')
    screenshot = pyscreenshot.grab(bbox=None, childprocess=None, backend=None)
    image = np.asarray(screenshot.convert('RGB'))
    screenshot.close()
    image = image[:, :, ::-1].copy()
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    print('detecting craters...')
    start = time.time()
    with graph.as_default():
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    boxes /= scale
    crater_count = 0
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < 0.05:
            break
        crater_count += 1
        b = box.astype(int)
        draw_box(draw, b, color=(255, 215, 0))
        caption = "{:.3f}".format(score)
        draw_caption(draw, b, caption)
    print(str(crater_count), ' craters detected!')
    fig = plt.figure(figsize=(15, 15), dpi=100)
    fig.figimage(draw, xo=0, yo=0)
    plt.show()


if __name__ == '__main__':
    main()
