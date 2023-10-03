<h1>Train Resnet with Custom dataset</h1>
<h2>I.How to run this repository<h2>
  <h3>1.For training</h3>
  <p>+Step0: Dowload data you need to classification, the struct of folder contain data to classification like this</p>
  <img src = "image.png"></img>
  <p>+Step1: Go to "data" folder then create another file .yaml like dog_cat.yaml</p>
  <p>+Step2: Open CMD, terminal, anaconda prompt</p>
  <p>+Step3: Clone this repository<p>
  <pre>git clone https://github.com/LuongTuanAnh163002/Resnet_AnhLT.git</pre>
  <p>+Step4: Move to the Resnet_AnhLT folder</p>
  <pre>cd Resnet_AnhLT</pre>
  <p>+Step5: Install all packges need</p>
  <pre>pip install -r requirements.txt</pre>
  <p>+Step6: Run the code below to training for pretrain</p>
  <pre>python train.py --model_type [resnet18, resnet34, resnet50, resnet101, resnet152] --pretrained --freeze --data file_name.yaml --epochs 50 --device 0</pre>
  <p>+Step7: Run the code below to training for weight initialize</p>
  <pre>python train.py --model_type [resnet18, resnet34, resnet50, resnet101, resnet152] --data file_name.yaml --epochs 50 --device 0</pre>
  <p>+Step8: Run the code below to training for using weight from previous train</p>
  <pre>python train.py --model_type [resnet18, resnet34, resnet50, resnet101, resnet152] --weight_init [file_name.pt, file_name.pth] --data file_name.yaml --epochs 50 --device 0</pre>
  <p>After you run and done training, all results save in runs/train/exp/..., folder runs automatic create after training done:</p>

  <h3>2.For detect</h3>
  <p>+Detect for file</p>
  <pre>python detect.py --source file_name.jpg --weights ../runs/train/../weights/__.pth --device 0</pre>
  <p>+Detect for folder</p>
  <pre>python detect.py --source path_folder --weights ../runs/train/../weights/__.pth --device 0</pre>

<h2>II.Try with example in google colab</h2>
<a href="https://colab.research.google.com/drive/12FKf4q1szLpT-tTUpl8mslSj9g6BhlYW?usp=sharing" rel="nofollow"><img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;"></a>
