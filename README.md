# Mnist-GAN-API
This is a simple GAN to generate handwritten numbers based on famous Mnist Dataset, a project mission for USTC
Python and Deep Learning course.

The code is test on an RTX2080ti GPU workstation, with a Intel-i7 8750 CPU, 32GB RAM and Ubuntu16.04 system.
The code built on an environment with Python3.6 and CUDA10, with PyTorch structure.
The Dataset comes from a popular handwritten number dataset Mnist, built by LeCun et al. http://yann.lecun.com/exdb/mnist/
The code utilizes 60,000 images from the Dataset to train the network.
The GUI interface is designed with pyqt5 package and Qtdesigner.

Guidance:
  open terminal and go to the right path
  
   $python3 ./GUI.py (GUI interface presents)
  
  Train Menu: Click 'Start Training' to start training the network. Click 'Stop Training' to stop
  
  Setting Menu: Adapt the network settings such as learning rate, batch size, max epoch etc. The change will be showed
      on the 'Recent Settings' label. (All settings cannot be changed when the network training starts or pauses.)
  
  'Pause/Resume' Button: Click the button the training will be suspended. Click again and it will resume. 
      You cannnot start a new training process now but stop this process first.
      
  'Set to Default' Button: Click the button, and all settings will go back to default ones showed on the 'Default Settings' 
      label above. (All settings cannot be set to default value when the network training starts or pauses.)
  
  During the training, some of the genearted image will be showed under the 'Results' and 'Epoch = **' labels.
  
  After the training ends, use terminal and go to the right path.
  $ tensorboard --logdir=mm-dd_hh-MM-ss  
  (Click the URL and open it on browser, you will find some of the training results and information text.)
  
  Other information like learning rate will be showed on terminal.
  
  In addition, because of the large number of the dataset, the first time training seems start slowly because of the process
      of reading data. Please wait for a few minutes and do not click the 'Stop Training' or 'Pause/Resume'.



The network structure is like the following:

Generator: 
Linear(input_noise_size, 1024) -> ReLU -> BatchNorm -> Linear(1024,7*7*128) -> BatchNorm ->Unflatten(batchsize, 128, 7, 7) -> 
ConvTrans(128,64,kernel_size=4, stride=2, padding=1) -> ReLU -> BatchNorm -> 
ConvTrans(64, 1, kernel_size=4, stride=2, padding=1) -> ReLU -> Tanh

Discriminator:

Conv(1, 32, kernel_size=5, stride=1) -> LeakyReLU(0.01) -> MaxPool(2, stride=2) -> Conv(32, 64, kernel_size=5, stride=1) ->
LeakyReLU(0.01) ->  MaxPool(2, stride=2) -> Linear(4*4*64, 4*64) -> LeakyReLU(0.01) -> Linear(4*64, 1) -> Sigmoid

The training loss is classical GAN BCELoss.

The usual problem for this network is Mode Collaps especially when the epoch is too large.

