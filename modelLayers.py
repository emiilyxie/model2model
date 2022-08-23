from transformers import pipeline
from min_dalle import MinDalle
import numpy
import torch
from PIL import Image
import requests
import jax
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, \
GPT2Tokenizer, GPT2Model, M2M100ForConditionalGeneration, M2M100Tokenizer
from enum import Enum
import uuid

class PretrainedModelLayer():
  inputType = None
  outputType = None

  def __init__(self):
    pass
  def makePrediction(self, input):
    pass
  def isInputValid(input):
    pass

class DataType(Enum):
  IMG = 1
  TEXT = 2

class GPT2Layer(PretrainedModelLayer):

  inputType = [DataType.TEXT]
  outputType = [DataType.TEXT]

  def __init__(self):
    super().__init__()
    self.model = pipeline('text-generation', model='gpt2')
  def makePrediction(self, input):
    if self.isInputValid(input):
      output = self.model(input, max_length=15, num_return_sequences=1)
      return output[0]['generated_text']
    else:
      raise Exception("Input to GPT2 must be a string.")
  def isInputValid(self, input):
      return type(input) == str

class DalleLayer(PretrainedModelLayer):

  inputType = [DataType.TEXT]
  outputType = [DataType.IMG]

  def __init__(self):
    super().__init__()
    self.model = MinDalle(
        models_root='./pretrained',
        dtype=torch.float32,
        is_mega=False, 
        is_reusable=True
      )
  def makePrediction(self, input):
    if self.isInputValid(input):
      image = self.model.generate_image(
          text=input,
          seed=-1,
          grid_size=1,
          is_verbose=True
      )
      imgArray = numpy.array(image)
      imgFile = Image.fromarray(imgArray)
      imgFile.save(f'{uuid.uuid1()}.png')
      return imgArray
    else:
      raise Exception("Input to Dalle must be a string.")
  def isInputValid(self, input):
    return type(input) == str

class VITGPT2Layer(PretrainedModelLayer):

  inputType = [DataType.IMG]
  outputType = [DataType.TEXT]

  def __init__(self):
    super().__init__()
    self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
  def makePrediction(self, input):
    if self.isInputValid(input):
      return self.predict_step(input)[0]
    else:
      raise Exception("Input to VITGPT2 must be an numpy array of shape (256,256,3).")
  def isInputValid(self, input):
    return type(input) == numpy.ndarray and input.shape == (256,256,3)
  def predict_step(self, input):
    gen_kwargs = {"max_length": 16, "num_beams": 4}
    pixel_values = self.feature_extractor(images=input, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(self.device)
    output_ids = self.model.generate(pixel_values, **gen_kwargs)

    preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

