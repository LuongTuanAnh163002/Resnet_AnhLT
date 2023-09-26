<h1>How to run this repository<h1>
  <h2>For training</h2>
  <h3>+Step1: Go to data/ folder then create another file .yaml like dog_cat.yaml</h3>
  <h3>+Step2: Open CMD, terminal, anaconda prompt</h3>
  <h3>+Step3: Clone this repository<h3>
  <pre>git clone https://github.com/LuongTuanAnh163002/Resnet_AnhLT.git</pre>
  <h3>+Step4: Move to the Web_Sign_Language_Detection folder</h3>
  <pre>cd Resnet_AnhLT</pre>
  <h3>+Step5: Install all packges need</h3>
  <pre>pip install -r requirements.txt</pre>
  <h3>+Step6: Run the code below to training for pretrain</h3>
  <pre>python train.py --model_type [resnet18, resnet34, resnet50, resnet101, resnet152] --pretrained --freeze --data file_name.yaml --epochs 50 --device 0</pre>
  <h3>+Step7: Run the code below to training for weight initialize</h3>
  <pre>python train.py --model_type [resnet18, resnet34, resnet50, resnet101, resnet152] --data file_name.yaml --epochs 50 --device 0</pre>
  <h3>+Step8: Run the code below to training for using weight from previous train</h3>
  <pre>python train.py --model_type [resnet18, resnet34, resnet50, resnet101, resnet152] --weight_init [file_name.pt, file_name.pth] --data file_name.yaml --epochs 50 --device 0</pre>
  <p>After you run and done training, all results save in runs/train/exp/..., folder runs automatic create after training done:</p>

  <h2>For detect</h2>
  <h3>+Detect for file</h3>
  <pre>python detect.py --source file_name.jpg --weights ../runs/train/../weights/__.pth --device 0</pre>
  <h3>+Detect for folder</h3>
  <pre>python detect.py --source path_folder --weights ../runs/train/../weights/__.pth --device 0</pre>
