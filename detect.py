from model.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from model.model_fc import Dc_model
import os
import cv2
from PIL import Image
from torch import nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import CustomImageDataset
from torch.utils.data import DataLoader
import argparse
from utils.generals import increment_path
from utils.generals import image_convert
from pathlib import Path
import time

def detect(opt):
  save_dir, source, weights, device_number = Path(opt.save_dir), opt.source, opt.weights,\
                                              opt.device
  
  save_dir.mkdir(parents=True, exist_ok=True)
  imgsz = opt.imgsz
  batch_size = 8
  detect_transforms = transforms.Compose([
                                  transforms.Resize((imgsz,imgsz)),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.5])])
  format_file = ["png", "jpg", "jpeg", "bmp"]
  isfile = os.path.isfile(source)
  isdir = os.path.isdir(source)
  if isfile:
    if source.split(".")[-1] in format_file:
      img = cv2.imread(source)
      file_name = source.split("/")[-1]
      img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      img_pil = detect_transforms(img_pil)
      img_pil = img_pil.unsqueeze(0)
    else:
      print("Format file is incorrect")
  if isdir:
    dataset = CustomImageDataset(source, detect_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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

  t0 = time.time()
  if isfile:
    print("Predicting for one image")
    model.eval()
    img_pil = img_pil.to(device)
    img_out = model.forward(img_pil)
    value, index_val = torch.max(img_out, 1)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.set_title('pred:{}, score:{}'.format(label_dict[index_val[0]], value[0]))
    path_to_save = os.path.join(save_dir, file_name)
    plt.savefig(path_to_save)
     
  elif isdir:
    print("Predicting image in folder")
    model.eval()
    cnt = 0
    for images in dataloader:
      if cnt == 3:
        break
      images = images.to(device)
      img_out = model.forward(images)
      value, index_val = torch.max(img_out, 1)
      fig = plt.figure(figsize=(35,9))
      for idx in np.arange(8):
          ax = fig.add_subplot(2,4,idx+1)
          ax.imshow(image_convert(images[idx])) 
          pred_label = index_val[idx]
          ax.set_title('pred:{}, score:{}'.format(label_dict[pred_label], value[idx]))
      file_name_batch = "batch" + str(cnt)
      path_to_save_batch = os.path.join(save_dir, file_name_batch)
      plt.savefig(path_to_save_batch)
      cnt += 1
  print(f"Results saved to {save_dir}")
  print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', default='', help='path to image or folder include image')
  parser.add_argument('--weights', default='', help='weight to detect image')
  parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
  parser.add_argument('--imgsz', type=int, default=64, help="Image size of detect image")
  parser.add_argument('--project', default='runs/detect', help='save to project/name')
  parser.add_argument('--name', default='exp', help='save to project/name')
  opt = parser.parse_args()
  opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
  detect(opt)
