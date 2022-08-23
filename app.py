from crypt import methods
from socket import socket
import modelLayers as mls
from flask import Flask, request, jsonify, stream_with_context, Response
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

layers = []

def init_server():
  gpt2layer = mls.GPT2Layer()
  dalleLayer = mls.DalleLayer()
  vitGpt2Layer = mls.VITGPT2Layer()
  layers = [gpt2layer, dalleLayer, vitGpt2Layer]
  return layers

def init_server_fast():
  gpt2layer = mls.GPT2Layer()
  layers = [gpt2layer]
  return layers

class SampleModel():
  def __init__(self, layers, layerIndices):
    self.layers = layers
    self.layerIndices = layerIndices
  def predict(self, inputs):
    x = inputs[0]
    for layerIndex in self.layerIndices:
      layer = self.layers[layerIndex]
      x = layer.makePrediction(x)
      emit('model output', {'data': x})

@app.route('/')
def index():
  return "hello world :)"

@socketio.on('process data')
def process_data(data):
  global layers
  if not layers:
    emit('server action', 'initializing layers')
    layers = init_server_fast()
    emit('server action', 'finished initializing layers')
  emit('server action', 'starting data processing')
  sampleModel = SampleModel(layers, data['modelIndices'])
  sampleModel.predict(data['inputs'])


# @socketio.on('after connect')
# def test_connect(data):
#   print(data)
#   emit('server response', {'data': 'Hello world'})

'''
@app.route('/stream', methods=['POST'])
def streamed_response():
    @stream_with_context
    def generate():
        yield 'Hello '
        time.sleep(3)
        yield 'World'
        time.sleep(1)
        yield '!'
    return Response(generate())

@app.route('/test', methods=['POST'])
def generateTest():
  def testText():
    gpt2layer = layerDict.get('gpt2layer', mls.GPT2Layer()) 
    #dalleLayer = mls.DalleLayer()
    #vitGpt2Layer = mls.VITGPT2Layer()
    #layers = [gpt2layer, dalleLayer, vitGpt2Layer]
    layers = [gpt2layer]
    input = "hello world"
    layerIndices = [0]
    sampleModel = mls.SampleModel(layers, layerIndices)
    output = sampleModel.predict(input)
    yield jsonify({'input': input})
    yield jsonify({'output': output})
  return Response(testText())
'''

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', debug=True)

'''
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import modelLayers

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)

hub_url = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
content_image_path = 'img/jaye-crop.jpg'
style_image_path = 'img/surtr-crop.jpg'
IMAGE_SHAPE = (256,256)

content_image = plt.imread(content_image_path)
style_image = plt.imread(style_image_path)
content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
content_image = tf.image.resize(content_image, IMAGE_SHAPE)
style_image = tf.image.resize(style_image, IMAGE_SHAPE)

module1 = hub.load(hub_url)
model = tf.Sequential([module1])

#outputs = hub_module(tf.constant(content_image), tf.constant(style_image))

#im = tensor_to_image(outputs[0])
#im.save('img/jaye-surtr.jpg')
# for seeing results quickly, run 
# docker run -v $(pwd)/img:/app/img pythoncode
# basically just maps my img directory to the docker img directory
'''