import os
import torch
import argparse
from pathlib import Path
from model.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from model.model_fc import Dc_model
import onnx

def export(opt):
  weights, device_number, imgsz = opt.weights, opt.device, opt.imgsz
  save_dir = "weights/"
  Path(save_dir).mkdir(parents=True, exist_ok=True)
  device = torch.device('cuda:'+ device_number if torch.cuda.is_available() else 'cpu')
  if device.type == "cpu":
    dict_weight = torch.load(weights, map_location=torch.device('cpu'))
  else:
    dict_weight = torch.load(weights)
  label_dict = dict_weight["classes"]
  model_type = dict_weight["model_type"]
  if model_type == "resnet18":
    model = resnet18()
    model = model.to(device)
    model_ = Dc_model(len(label_dict)).to(device)
    model.fc = model_
    model.load_state_dict(dict_weight['model_state_dict'])
  
  elif model_type == "resnet34":
    model = resnet34()
    model = model.to(device)
    model_ = Dc_model(len(label_dict)).to(device)
    model.fc = model_
    model.load_state_dict(dict_weight['model_state_dict'])

  elif model_type == "resnet50":
    model = resnet50()
    model = model.to(device)
    model_ = Dc_model(len(label_dict)).to(device)
    model.fc = model_
    model.load_state_dict(dict_weight['model_state_dict'])

  elif model_type == "resnet101":
    model = resnet101()
    model = model.to(device)
    model_ = Dc_model(len(label_dict)).to(device)
    model.fc = model_
    model.load_state_dict(dict_weight['model_state_dict'])

  elif model_type == "resnet152":
    model = resnet152()
    model = model.to(device)
    model_ = Dc_model(len(label_dict)).to(device)
    model.fc = model_
    model.load_state_dict(dict_weight['model_state_dict'])
  
  model.eval()
  inputs = torch.randn(1, 3, imgsz, imgsz)
  print("begin to convert onnx")
  
  torch.onnx.export(model,
                  inputs,
                  'weights/{}_{}x{}.onnx'.format(model_type, imgsz, imgsz),
                  opset_version=11,
                  input_names=['input'],
		  output_names=['classification'])
  print("ONXX done")
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--weights', default='', help='weight to detect image')
  parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
  parser.add_argument('--imgsz', type=int, default=64, help="Image size of detect image")
  opt = parser.parse_args()
  export(opt)
