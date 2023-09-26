import yaml
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from  torch import optim
from torchvision import transforms, utils, models
import torch.utils.model_zoo as model_zoo
from tqdm import tqdm
from collections import OrderedDict
import argparse
from model.model_fc import Dc_model
from utils.generals import increment_path
from utils.metrics import ConfusionMatrix, draw_confusionmatrix, compute_f1_pr_rc
from model.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import os
import time

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def train(opt, tb_writer):
  save_dir, epochs, batch_size, weight_init, device_number, imgsz= \
              Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights_init,opt.device, opt.imgsz
  model_type = opt.model_type
  metric_val = opt.metric
  wdir = save_dir / 'weights'
  wdir.mkdir(parents=True, exist_ok=True)  # make dir
  last = wdir / 'last.pth' #save last weight 
  best = wdir / 'best.pth' #save best weight
  device = torch.device('cuda:'+ device_number if torch.cuda.is_available() else 'cpu')
  with open(opt.data) as f:
      data_dict = yaml.load(f, Loader=yaml.SafeLoader)
  nc = int(data_dict['nc'])  # number of classes
  names = data_dict['names']

  #----------------------------load model-------------------------------------
  initialize = weight_init.endswith('.pt') or weight_init.endswith('.pth')
  
  if initialize:
    if model_type == "resnet18":
      model = resnet18()
      model = model.to(device)
      model_ = Dc_model(nc).to(device)
      model.fc = model_

    elif model_type == "resnet34":
      model = resnet34()
      model = model.to(device)
      model_ = Dc_model(nc).to(device)
      model.fc = model_
      
    elif model_type == "resnet50":
      model = resnet50()
      model = model.to(device)
      model_ = Dc_model(nc).to(device)
      model.fc = model_
      
    elif model_type == "resnet101":
      model = resnet101()
      model = model.to(device)
      model_ = Dc_model(nc).to(device)
      model.fc = model_
      
    elif model_type == "resnet152":
      model = resnet152()
      model = model.to(device)
      model_ = Dc_model(nc).to(device)
      model.fc = model_
      
    if device.type == "cpu":
      model.load_state_dict(torch.load(weight_init, map_location=torch.device('cpu'))['model_state_dict'])
    else:
      model.load_state_dict(torch.load(weight_init)['model_state_dict'])
    
    print("Initilize weight for " + model_type + " archirtech")
    print("Initilize weight from", weight_init)
  else:
    if opt.pretrained == True:
      if model_type == "resnet18":
        model = resnet18()
        model_ = Dc_model(nc).to(device)
        if os.path.exists('resnet18-5c106cde.pth'):
          model.load_state_dict(torch.load('resnet18-5c106cde.pth'))
          model = model.to(device)
          model.fc = model_
          print("Find file weight exist, using file for resnet18 in transfer leanring")
        else:
          model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
          torch.save(model.state_dict(), "resnet18-5c106cde.pth")
          model = model.to(device)
          model.fc = model_
          print("File not found, dowloading from:", model_urls['resnet18'])

      elif model_type == "resnet34":
        model = resnet34()
        model_ = Dc_model(nc).to(device)
        if os.path.exists('resnet34-333f7ec4.pth'):
          model.load_state_dict(torch.load('resnet34-333f7ec4.pth'))
          model = model.to(device)
          model.fc = model_
          print("Find file weight exist, using file for resnet34 in transfer leanring")
        else:
          model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
          torch.save(model.state_dict(), "resnet34-333f7ec4.pth")
          model = model.to(device)
          model.fc = model_
          print("File not found, dowloading from:", model_urls['resnet34'])
      elif model_type == "resnet50":
        model = resnet50()
        model_ = Dc_model(nc).to(device)
        if os.path.exists('resnet50-19c8e357.pth'):
          model.load_state_dict(torch.load('resnet50-19c8e357.pth'))
          model = model.to(device)
          model.fc = model_
          print("Find file weight exist, using file for resnet50 in transfer leanring")
        else:
          model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
          torch.save(model.state_dict(), "resnet50-19c8e357.pth")
          model = model.to(device)
          model.fc = model_
          print("File not found, dowloading from:", model_urls['resnet50'])

      elif model_type == "resnet101":
        model = resnet101()
        model_ = Dc_model(nc).to(device)
        if os.path.exists('resnet101-5d3b4d8f.pth'):
          model.load_state_dict(torch.load('resnet101-5d3b4d8f.pth'))
          model = model.to(device)
          model.fc = model_
          print("Find file weight exist, using file for resnet101 in transfer leanring")
        else:
          model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
          torch.save(model.state_dict(), "resnet101-5d3b4d8f.pth")
          model = model.to(device)
          model.fc = model_
          print("File not found, dowloading from:", model_urls['resnet101'])

      elif model_type == "resnet152":
        model = resnet152()
        model_ = Dc_model(nc).to(device)
        if os.path.exists('resnet152-b121ed2d.pth'):
          model.load_state_dict(torch.load('resnet152-b121ed2d.pth'))
          model = model.to(device)
          model.fc = model_
          print("Find file weight exist, using file for resnet152 in transfer leanring")
        else:
          model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
          torch.save(model.state_dict(), "resnet101-5d3b4d8f.pth")
          model = model.to(device)
          model.fc = model_
          print("File not found, dowloading from:", model_urls['resnet152'])
          
      print("Initilize weight for " + model_type + " using tranfer learning")
    
    else:
      if model_type == "resnet18":
        model = resnet18()
        model = model.to(device)
        model_ = Dc_model(nc).to(device)
        model.fc = model_

      if model_type == "resnet34":
        model = resnet34()
        model = model.to(device)
        model_ = Dc_model(nc).to(device)
        model.fc = model_

      if model_type == "resnet50":
        model = resnet50()
        model = model.to(device)
        model_ = Dc_model(nc).to(device)
        model.fc = model_

      if model_type == "resnet101":
        model = resnet101()
        model = model.to(device)
        model_ = Dc_model(nc).to(device)
        model.fc = model_

      if model_type == "resnet152":
        model = resnet152()
        model = model.to(device)
        model_ = Dc_model(nc).to(device)
        model.fc = model_
      print("Initilize weight random for", model_type)
  #----------------------------load model-------------------------------------
  
  #-------------------------load train_test dataset-----------------------------------
  train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                transforms.RandomHorizontalFlip(),
                                transforms.Resize((imgsz,imgsz)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])])
  test_transforms = transforms.Compose([
                                  transforms.Resize((imgsz,imgsz)),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.5])])
  
  train_path = data_dict['train']
  test_path = data_dict['val']
  
  train_data = datasets.ImageFolder(train_path, transform=train_transforms)
  test_data = datasets.ImageFolder(test_path, transform=test_transforms)
  
  trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
  testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,shuffle=True)
  #-------------------------load train_test dataset-----------------------------------
  
  #-----------------------add tensorboard histogram---------------------------
  iter_ = iter(trainloader)
  image,label = next(iter_)
  if tb_writer:
    tb_writer.add_histogram('classes', label, 0)
  #-----------------------add tensorboard histogram---------------------------
  if opt.freeze:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    print("Freeze for transfer learning")
  else:
    for param in model.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
  
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(),lr=0.001)
  t0 = time.time()
  train_loss = []
  val_loss = []
  val_acc = []
  torch.save(model.state_dict(), wdir / 'init.pth')
  best_accuracy = 0
  print(f'Logging results to {save_dir}\n'
        f'Starting training for {epochs} epochs...')
  #----------------------------start epoch------------------------------------
  for epoch in range(epochs):
    running_loss = 0.0
    running_score = 0.0
#       model.train()
    print(('\n' + '%15s' * 4) % ('Epoch', 'gpu_mem', 'train_loss', 'accuracy'))
    pbar = tqdm(trainloader)
    #-----------------------------start training-------------------------------------------
    for image,label in pbar:
      image = image.to(device)
      label = label.to(device)
      optimizer.zero_grad()
      y_pred = model.forward(image)
      loss = criterion(y_pred,label)         
      loss.backward() #calculate derivatives 
      optimizer.step() # update parameters
      val, index_ = torch.max(y_pred,axis=1)
      running_score += torch.sum(index_ == label.data).item()
      running_loss += loss.item()
      running_epoch_score = running_score/len(trainloader.dataset)
      running_epoch_loss = running_loss/len(trainloader.dataset)
      mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
      s = ('%15s' * 2 + '%15.4g' * 2) % ('%g/%g' % (epoch, epochs - 1), mem, running_epoch_loss, running_epoch_score)
      pbar.set_description(s)
    epoch_score = running_score/len(trainloader.dataset)
    epoch_loss = running_loss/len(trainloader.dataset)

    #-------------------add tensorboard scalar------------------
    if tb_writer:
      tb_writer.add_scalar("train/loss", epoch_loss, epoch)
      tb_writer.add_scalar("train/accuracy", epoch_score, epoch)
    #-------------------add tensorboard scalar------------------

    train_loss.append(epoch_loss)
    # print("Training loss: {}, accuracy: {}".format(epoch_loss,epoch_score))
    #--------------------------end training------------------------------------------------
    with torch.no_grad():
      model.eval()
      running_loss = 0.0
      running_score = 0.0
      s = ('%20s' + '%12s' * 1) % ('Val_loss', 'accuracy')
      for image,label in tqdm(testloader, s):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(image)
            loss = criterion(y_pred,label)
            running_loss += loss.item()

            val, index_ = torch.max(y_pred,axis=1)
            running_score += torch.sum(index_ == label.data).item()
      
      epoch_score = running_score/len(testloader.dataset)
      epoch_loss = running_loss/len(testloader.dataset)

      #-------------------add tensorboard scalar------------------
      if tb_writer:
        tb_writer.add_scalar("val/loss", epoch_loss, epoch)
        tb_writer.add_scalar("val/accuracy", epoch_score, epoch)
      #-------------------add tensorboard scalar------------------

      if epoch_score > best_accuracy:
        best_accuracy = epoch_score
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': epoch_score,
            'classes': names,
            'model_type': model_type,
        }
        torch.save(ckpt, best)
      pf = '\n%20.4g' + '%12s' * 1
      val_loss.append(epoch_loss)
      val_acc.append(epoch_score)
      print(pf % (round(epoch_loss,3), round(epoch_score,3)))
      # print("Validation loss: {}, accuracy: {}".format(epoch_loss,epoch_score))
  #--------------------------end epoch------------------------------------------------
  print('%g epochs completed in %.3f hours.\n' % (epochs, (time.time() - t0) / 3600))
  pr, rc, f1 = compute_f1_pr_rc(model, testloader, device, metric_val)
  plt.bar([0, 1, 2], [pr, rc, f1], color=["green", "blue", "black"])
  plt.xticks([0,1,2], ["Precision", "Recall", "F1_score"])
  plt.savefig(save_dir/"pr_rc_f1.png")
  print("Precision score:", pr)
  print("Recall score:", rc)
  print("F1 score:", f1)
  
  #-------------------------save confusion_matrix-------------------------------------
  confusion_mat = ConfusionMatrix(nc, model, device, testloader)
  draw_confusionmatrix(confusion_mat, names, save_dir)
  #-------------------------save confusion_matrix-------------------------------------
  
  #-------------------------save loss_curve-------------------------------------------
  plt.figure(figsize=(7, 7))
  epochs_lst = list(range(len(train_loss)))
  plt.plot(epochs_lst, train_loss, label = "train_loss")
  plt.plot(epochs_lst, val_loss, label = "val_loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend(loc="upper right")
  path_to_save = save_dir/"Loss_curve.png"
  plt.savefig(path_to_save)
  #-------------------------save loss_curve-------------------------------------------
  ckpt_last = {
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': val_acc[-1],
            'classes': names,
            'model_type':model_type,
        }
  torch.save(ckpt_last, last)
  print("Optimizer stripped from ", best)
  print("Optimizer stripped from ", last)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='resnet18', help='type of model: resnet18, resnet34, resnet50, resnet101, resnet152')
    parser.add_argument('--pretrained', action='store_true', help='initial weights transfer path')
    parser.add_argument('--weights_init', type=str, default='', help='initial weights path')
    parser.add_argument('--data', type=str, default='', help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', type=int, default=64, help='Size of image in training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--freeze', action='store_true', help='Freeze some first layer of pretrain')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--metric', default='macro', help='evaluate precision, recall, f1')
    opt = parser.parse_args()
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
    tb_writer = SummaryWriter(opt.save_dir)
    print(f"Start with 'tensorboard --logdir {opt.save_dir}', view at http://localhost:6006/")
    train(opt, tb_writer)