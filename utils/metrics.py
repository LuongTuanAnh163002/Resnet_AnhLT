import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

def ConfusionMatrix(nb_classes, model, device, dataloader):
  confusion_matrix = np.zeros((nb_classes, nb_classes))
  with torch.no_grad():
      for i, (inputs, classes) in enumerate(dataloader):
          inputs = inputs.to(device)
          classes = classes.to(device)
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)
          for t, p in zip(classes.view(-1), preds.view(-1)):
                  confusion_matrix[t.long(), p.long()] += 1
  return confusion_matrix

def draw_confusionmatrix(confusion_matrix, class_names, parent_folder_save):
  plt.figure(figsize=(7,7))
  df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
  heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
  heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
  heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  path_save = parent_folder_save / "confusion_matrix.png"
  plt.savefig(path_save)


def compute_f1_pr_rc(model, dataloader, device, type_compute):
    model.eval()
    iter_ = iter(dataloader)
    images,labels = next(iter_)
    images = images.to(device)
    img_out = model.forward(images)
    value, index_val = torch.max(img_out, 1)
    true_value = labels.numpy()
    pred_value = index_val.cpu().numpy()
    if type_compute == "binary":
      pr = precision_score(true_value, pred_value)
      rc = recall_score(true_value, pred_value)
      if pr == 0 and rc == 0:
        f1 = 0
      else:
        f1 = (2*pr*rc)/(pr+rc) 
      return pr, rc, f1
    
    elif type_compute == "macro":
      pr = precision_score(true_value, pred_value, average="macro")
      rc = recall_score(true_value, pred_value, average="macro")
      if pr == 0 and rc == 0:
        f1 = 0
      else:
        f1 = (2*pr*rc)/(pr+rc) 
      return pr, rc, f1
    
    elif type_compute == "micro":
      pr = precision_score(true_value, pred_value, average="micro")
      rc = recall_score(true_value, pred_value, average="micro")
      if pr == 0 and rc == 0:
        f1 = 0
      else:
        f1 = (2*pr*rc)/(pr+rc) 
      return pr, rc, f1