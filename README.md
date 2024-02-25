<h1>RESNET: Build from scratch, Train Resnet with Custom dataset</h1>
<div align="center" dir="auto">
<a href="https://github.com/LuongTuanAnh163002/Resnet_AnhLT/blob/main/LICENSE"><img src="https://camo.githubusercontent.com/00b6aa098f95cc8559f5f72a62f63261e44a1f09f0f560ca4c8ab25d4a631f05/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c6963656e73652d4d49542d253343434f4c4f522533452e7376673f7374796c653d666f722d7468652d6261646765" alt="Generic badge" data-canonical-src="https://img.shields.io/badge/License-MIT-%3CCOLOR%3E.svg?style=for-the-badge" style="max-width: 100%;"></a>
<a href="https://pytorch.org/get-started/locally/" rel="nofollow"><img src="https://camo.githubusercontent.com/0add0c0b6ec6267b61016063796469feb03cc17c93d9f04201e25d0f12651de0/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5059544f5243482d312e31302b2d7265643f7374796c653d666f722d7468652d6261646765266c6f676f3d7079746f726368" alt="PyTorch - Version" data-canonical-src="https://img.shields.io/badge/PYTORCH-1.10+-red?style=for-the-badge&amp;logo=pytorch" style="max-width: 100%;"></a>
<a href="https://www.python.org/downloads/" rel="nofollow"><img src="https://camo.githubusercontent.com/c2623d41ae89703a8d56dab2e458028b95b87d8ce1897ff29930ef267e9e77e0/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f505954484f4e2d332e372b2d7265643f7374796c653d666f722d7468652d6261646765266c6f676f3d707974686f6e266c6f676f436f6c6f723d7768697465" alt="Python - Version" data-canonical-src="https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&amp;logo=python&amp;logoColor=white" style="max-width: 100%;"></a>
<br></p>
</div>

<details open="">
  <summary>Table of Contents</summary>
  <ol dir="auto">
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#project-structure">Project Structure</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li><a href="#custom-dataset">How to run repository with custom dataset</a></li>
    <li><a href="#colab">Try with example in google colab</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#about-the-project">About The Project<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>

<p dir="auto">In this project we will build RESNET from scratch and training with all CUSTOM dataset</p>

<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#project-structure">Project Structure<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" dir="auto">
  <pre>Resnet_AnhLT
  │   train.py                      <span class="pl-c"><span class="pl-c">#</span> Train script</span>
  │   detect.py                     <span class="pl-c"><span class="pl-c">#</span> Detect script inference</span>
  
  ├───model
  │       model_fc.py               <span class="pl-c"><span class="pl-c">#</span>Transfer RESNET with custom dataset</span>
  │       resnet.py                 <span class="pl-c"><span class="pl-c">#</span> RESNET architecture</span>
  │
  ├───data
  │       dog_cat.yaml              <span class="pl-c"><span class="pl-c">#</span>Config data dog_cat example</span>
  │
  └───utils
      │   datasets.py               <span class="pl-c"><span class="pl-c">#</span>Processing datasets</span>
      │   metrics.py                <span class="pl-c"><span class="pl-c">#</span> Compute metrics</span>
      │   generals.py               <span class="pl-c"><span class="pl-c">#</span> Various helper functions</span>
  </pre>
</div>

<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#data-preparation">Data Preparation<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>

<pre>Resnet_AnhLT
└───datasets
    ├───train
    │   ├───name_class1
    │   └───name_class2
        .............
    ├───val
    │   ├───name_class1
    │   └───name_class2
        .............
    ├───test
    │   ├───name_class1
    │   └───name_class2
        ..............
</pre>

<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#custom-dataset">How to run repository with custom dataset<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>
  <h3>1.For training</h3>
  <p>+Step1: Dowload data you need to classification, the struct of folder contain data to classification like data preparation part</p>
  <p>+Step2: Open anaconda prompt and install virtual environment and all package</p>
  <pre>
  conda create --name resnet python=3.10.12
  git clone https://github.com/LuongTuanAnh163002/Resnet_AnhLT.git
  cd Resnet_AnhLT
  conda activate resnet
  pip install -r requirements.txt
  </pre>
  
  <p>+Step3: Go to "data" folder then create another file .yaml like dog_cat.yaml</p>
  <p>+Step4: Run the code below to training</p>
  <pre>
  #for pretrain
  python train.py --model_type [resnet18, resnet34, resnet50, resnet101, resnet152] --pretrained --freeze --data file_name.yaml --epochs 50 --device [0, 1, 2,..]
  
  #for weight initialize
  python train.py --model_type [resnet18, resnet34, resnet50, resnet101, resnet152] --data file_name.yaml --epochs 50 --device [0, 1, 2,..]
  
  #for using weight from previous train
  python train.py --model_type [resnet18, resnet34, resnet50, resnet101, resnet152] --weight_init [file_name.pt, file_name.pth] --data file_name.yaml --epochs 50 --device [0, 1, 2,..]
  </pre>
  <p>After you run and done training, all results save in runs/train/exp/..., folder runs automatic create after training done:</p>

  <h3>2.For detect</h3>
  <pre>
  #for file
  python detect.py --source file_name.jpg --weights ../runs/train/../weights/__.pth --device [0, 1, 2,..]
  #for folder
  python detect.py --source path_folder --weights ../runs/train/../weights/__.pth --device [0, 1, 2,..]
  </pre>

  <h3>3.For export</h3>
  <pre>python export.py --weights ../runs/train/../weights/__.pth --device [0, 1, 2,..]</pre>

  <h3>4.Launch tensorboard</h3>
  <pre>tensorboard --logdir ../runs/train/name_project --bind_all --port=2002</pre>

<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#colab">Try with example in google colab<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>
<a href="https://colab.research.google.com/drive/1AnL-yzia87tQGsob0AROPYM9bKAE_4UF?usp=sharing" rel="nofollow"><img src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;"></a>


<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#license">License<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>
<p dir="auto">See <code>LICENSE</code> for more information.</p>

<h2 tabindex="-1" id="user-content-about-the-project" dir="auto"><a class="heading-link" href="#acknowledgements">Acknowledgements<svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="m7.775 3.275 1.25-1.25a3.5 3.5 0 1 1 4.95 4.95l-2.5 2.5a3.5 3.5 0 0 1-4.95 0 .751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018 1.998 1.998 0 0 0 2.83 0l2.5-2.5a2.002 2.002 0 0 0-2.83-2.83l-1.25 1.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042Zm-4.69 9.64a1.998 1.998 0 0 0 2.83 0l1.25-1.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042l-1.25 1.25a3.5 3.5 0 1 1-4.95-4.95l2.5-2.5a3.5 3.5 0 0 1 4.95 0 .751.751 0 0 1-.018 1.042.751.751 0 0 1-1.042.018 1.998 1.998 0 0 0-2.83 0l-2.5 2.5a1.998 1.998 0 0 0 0 2.83Z"></path></svg></a></h2>

<ul dir="auto">
<li><a href="https://github.com/WongKinYiu/yolov7.git">YOLOv7</a></li>
<li><a href="https://github.com/pytorch/vision.git">ResNet torchvision</a></li>
</ul>
