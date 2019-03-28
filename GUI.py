# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MnistGAN.ui'
#
# Created by: PyQt5 UI code generator 5.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import torch

from load_data import load_data
import Net_Model
import torch.optim as optim
import torch.nn as nn
import numpy as np
from datetime import datetime
import os
from tensorboardX import SummaryWriter
import torchvision
import torchvision.utils as vutils
import imageio

'''
Parameters
'''
global netBatchSize
netBatchSize = 100

global netMaxEpoch
netMaxEpoch = 100

global netInput_Size
netInput_Size = 100

global netLR
netLR = 1e-4

global netoptimizerlabel
netoptimizerlabel = 'Adam'

global nettraininglabel
nettraininglabel= False

global netpauselabel
netpauselabel= False

global now_epoch
now_epoch =1

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_path = './Dataset/train/'

class TrainingThread(QtCore.QThread):
    stoptrigger = QtCore.pyqtSignal()

    def __init__(self):
        super(TrainingThread,self).__init__()

    def run(self):

        print('train')
        global netpauselabel
        global nettraininglabel
        global netoptimizerlabel
        global netLR
        global netMaxEpoch
        global netBatchSize
        global netInput_Size
        global train_loader
        print(netLR)
        print(nettraininglabel)
        print(netMaxEpoch)
        print(netBatchSize)
        print(netInput_Size)
        print(netoptimizerlabel)

        train_loader = load_data(train_path, batch_size=netBatchSize)

        Gnet = Net_Model.G(netInput_Size, netBatchSize).to(device)
        Gnet.initialize_weights()
        Dnet = Net_Model.D(netBatchSize).to(device)
        Dnet.initialize_weights()
        if netoptimizerlabel == 'Adam':
            optimizer_G = optim.Adam(Gnet.parameters(), lr=netLR, betas=(0.9, 0.999), eps=10e-8)
        elif netoptimizerlabel == 'Adagrad':
            optimizer_G = optim.Adagrad(Gnet.parameters(), lr=netLR)
        elif netoptimizerlabel == 'RMSProp':
            optimizer_G = optim.RMSprop(Gnet.parameters(), lr=netLR)
        if netoptimizerlabel == 'Adam':
            optimizer_D = optim.Adam(Dnet.parameters(), lr=netLR, betas=(0.9, 0.999), eps=10e-8)
        elif netoptimizerlabel == 'Adagrad':
            optimizer_D = optim.Adagrad(Dnet.parameters(), lr=netLR)
        elif netoptimizerlabel == 'RMSProp':
            optimizer_D = optim.RMSprop(Dnet.parameters(), lr=netLR)
        criterion = nn.BCELoss()
        now_time = datetime.now()
        time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
        log_dir = os.path.join('./Result/', time_str)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
        noise = np.random.randn(60000, netInput_Size, 1, 1)
        noise_test = np.random.randn(netBatchSize, netInput_Size, 1, 1)
        writer.add_text('Normalized input\n' + 'parameters:','dataset_num = ' + str(60000) + '\nlr_init_g = ' + str(netLR) +'\nlr_init_d = ' + str(netLR) + '\nNoise_dim = ' + str(netInput_Size) + '\n Batch_size = ' + str(netBatchSize)+'\noptimizer = '+ netoptimizerlabel)
        noise_test = torch.Tensor(noise_test).to(device)

        for epoch in range(netMaxEpoch):
            global now_epoch
            now_epoch = epoch+1
            for i, data in enumerate(train_loader, 0):
                noise_i = noise[int(i * netBatchSize):int((i + 1) * netBatchSize), :, :, :]
                noise_i = noise_i / noise_i.max()
                noise_i = torch.Tensor(noise_i).to(device)
                optimizer_D.zero_grad()
                inputs = data[0].to(device)
                output = Dnet(inputs).view(-1)
                b_size = output.size(0)
                label = torch.full((b_size,), 1, device=device)
                errorD_real = criterion(output, label)
                errorD_real.backward()
                # optimizer_G.zero_grad()
                D_x = output.mean().item()
                fake = Gnet(noise_i)
                label.fill_(0)
                output = Dnet(fake.detach()).view(-1)
                errorD_fake = criterion(output, label)
                errorD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errorD_real + errorD_fake
                optimizer_D.step()
                if i % 10 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tD(x): %.4f\tD(G(z)): %.4f '
                          % (epoch+1, netMaxEpoch, i, len(train_loader),
                             errD, D_x, D_G_z1))
                optimizer_D.step()
                noise_index = np.random.randint(noise.shape[0], size=60000)
                noise_s = noise[noise_index, :, :, :]

                if nettraininglabel == False:
                    writer.close()
                    print('stop')
                    if os.path.exists('./Result/show1') and  os.path.exists('./Result/show2') and os.path.exists('./Result/show3') and os.path.exists('./Result/show4'):
                        os.remove('./Result/show1')
                        os.remove('./Result/show2')
                        os.remove('./Result/show3')
                        os.remove('./Result/show4')
                    self.stoptrigger.emit()
                    break
                if netpauselabel == True:
                    while 1:
                        if nettraininglabel == False:
                            print('stop')
                            writer.close()
                            if os.path.exists('./Result/show1') and os.path.exists('./Result/show2') and os.path.exists(
                                    './Result/show3') and os.path.exists('./Result/show4'):
                                os.remove('./Result/show1')
                                os.remove('./Result/show2')
                                os.remove('./Result/show3')
                                os.remove('./Result/show4')
                            self.stoptrigger.emit()
                        if netpauselabel == False:
                            print('resume')
                            break

            for i in range(int(60000 / netBatchSize)):

                if nettraininglabel == False:
                    writer.close()
                    print('stop')
                    if os.path.exists('./Result/show1') and  os.path.exists('./Result/show2') and os.path.exists('./Result/show3') and os.path.exists('./Result/show4'):
                        os.remove('./Result/show1')
                        os.remove('./Result/show2')
                        os.remove('./Result/show3')
                        os.remove('./Result/show4')
                    self.stoptrigger.emit()
                    break
                if netpauselabel == True:
                    while 1:
                        if nettraininglabel == False:
                            print('stop')
                            writer.close()
                            if os.path.exists('./Result/show1') and os.path.exists('./Result/show2') and os.path.exists(
                                    './Result/show3') and os.path.exists('./Result/show4'):
                                os.remove('./Result/show1')
                                os.remove('./Result/show2')
                                os.remove('./Result/show3')
                                os.remove('./Result/show4')
                            self.stoptrigger.emit()
                        if netpauselabel == False:
                            print('resume')
                            break

                optimizer_G.zero_grad()
                noise_i = noise_s[int(i * netBatchSize):int((i + 1) * netBatchSize), :, :, :]
                noise_i = noise_i / noise_i.max()
                noise_i = torch.Tensor(noise_i).to(device)
                fake = Gnet(noise_i)
                label.fill_(1)
                output = Dnet(fake).view(-1)
                errorG = criterion(output, label)
                errorG.backward()
                D_G_z2 = output.mean().item()
                optimizer_G.step()
                if i % 10 == 0:
                    print('[%d/%d][%d/%d]\tErrorG:%.4f\tD(G(z)): %.4f ' % (
                    epoch+1, netMaxEpoch, i, 60000 / netBatchSize, errorG, D_G_z2))

            if nettraininglabel == False:
                writer.close()
                print('stop')
                if os.path.exists('./Result/show1') and os.path.exists('./Result/show2') and os.path.exists(
                        './Result/show3') and os.path.exists('./Result/show4'):
                    os.remove('./Result/show1')
                    os.remove('./Result/show2')
                    os.remove('./Result/show3')
                    os.remove('./Result/show4')
                self.stoptrigger.emit()
                break
            if netpauselabel == True:
                while 1:
                    if nettraininglabel == False:
                        print('stop')
                        writer.close()
                        if os.path.exists('./Result/show1') and os.path.exists('./Result/show2') and os.path.exists(
                                './Result/show3') and os.path.exists('./Result/show4'):
                            os.remove('./Result/show1')
                            os.remove('./Result/show2')
                            os.remove('./Result/show3')
                            os.remove('./Result/show4')
                        self.stoptrigger.emit()
                    if netpauselabel == False:
                        print('resume')
                        break

            with torch.no_grad():
                fake = Gnet(noise_test)
                fake_cpu = fake.to('cpu')
                show = fake_cpu.numpy()

                show1 = show[np.random.randint(0,netBatchSize), 0, :, :]
                show2 = show[np.random.randint(0,netBatchSize), 0, :, :]
                show3 = show[np.random.randint(0,netBatchSize), 0, :, :]
                show4 = show[np.random.randint(0,netBatchSize), 0, :, :]
                if not os.path.exists('./Result'):
                    os.makedirs('./Result')
                imageio.imwrite('./Result/show1', show1, '.png')
                imageio.imwrite('./Result/show2', show2, '.png')
                imageio.imwrite('./Result/show3', show3, '.png')
                imageio.imwrite('./Result/show4', show4, '.png')

                fake = vutils.make_grid(fake, normalize=True, scale_each=True)

            if (epoch ==0)or(epoch+1) % (netMaxEpoch/10) == 0:
                writer.add_image('Image', fake, epoch+1)
            if nettraininglabel == False:
                print('stop')
                writer.close()
                if os.path.exists('./Result/show1') and os.path.exists('./Result/show2') and os.path.exists(
                        './Result/show3') and os.path.exists('./Result/show4'):
                    os.remove('./Result/show1')
                    os.remove('./Result/show2')
                    os.remove('./Result/show3')
                    os.remove('./Result/show4')
                self.stoptrigger.emit()
                break
            if netpauselabel == True:
                 while 1:
                    if nettraininglabel == False:
                        print('stop')
                        writer.close()
                        if os.path.exists('./Result/show1') and os.path.exists('./Result/show2') and os.path.exists(
                                './Result/show3') and os.path.exists('./Result/show4'):
                            os.remove('./Result/show1')
                            os.remove('./Result/show2')
                            os.remove('./Result/show3')
                            os.remove('./Result/show4')
                        self.stoptrigger.emit()
                    if netpauselabel == False:
                        print('resume')
                        break

        writer.close()
        print('stop')
        if os.path.exists('./Result/show1') and os.path.exists('./Result/show2') and os.path.exists(
                './Result/show3') and os.path.exists('./Result/show4'):
            os.remove('./Result/show1')
            os.remove('./Result/show2')
            os.remove('./Result/show3')
            os.remove('./Result/show4')
        self.stoptrigger.emit()

class Updateresult(QtCore.QThread):
    trigger = QtCore.pyqtSignal()
    def __init__(self):
        super(Updateresult,self).__init__()
    def run(self):
        while 1:
            if nettraininglabel == True:
                self.trigger.emit()
                self.sleep(1)



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(599, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(40, 480, 511, 71))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pausebutton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pausebutton.setObjectName("pausebutton")
        self.buttonGroup = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.pausebutton)
        self.horizontalLayout.addWidget(self.pausebutton)
        self.setdefaultbutton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.setdefaultbutton.setObjectName("setdefaultbutton")
        self.horizontalLayout.addWidget(self.setdefaultbutton)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(56, 7, 491, 71))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setItalic(True)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(410, 90, 131, 151))
        self.label_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_4.setObjectName("label_4")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(130, 60, 191, 221))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setSpacing(1)
        self.gridLayout.setObjectName("gridLayout")
        self.label_15 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_15.sizePolicy().hasHeightForWidth())
        self.label_15.setSizePolicy(sizePolicy)
        self.label_15.setBaseSize(QtCore.QSize(0, 0))
        self.label_15.setText("")
        self.label_15.setPixmap(QtGui.QPixmap("./Dataset/train/train_151.png"))
        self.label_15.setScaledContents(False)
        self.label_15.setObjectName("label_15")
        self.gridLayout.addWidget(self.label_15, 2, 4, 1, 1)
        self.label_27 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_27.sizePolicy().hasHeightForWidth())
        self.label_27.setSizePolicy(sizePolicy)
        self.label_27.setBaseSize(QtCore.QSize(0, 0))
        self.label_27.setText("")
        self.label_27.setPixmap(QtGui.QPixmap("./Dataset/train/train_426.png"))
        self.label_27.setScaledContents(False)
        self.label_27.setObjectName("label_27")
        self.gridLayout.addWidget(self.label_27, 3, 3, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_23.sizePolicy().hasHeightForWidth())
        self.label_23.setSizePolicy(sizePolicy)
        self.label_23.setBaseSize(QtCore.QSize(0, 0))
        self.label_23.setText("")
        self.label_23.setPixmap(QtGui.QPixmap("./Dataset/train/train_98.png"))
        self.label_23.setScaledContents(False)
        self.label_23.setObjectName("label_23")
        self.gridLayout.addWidget(self.label_23, 2, 3, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setBaseSize(QtCore.QSize(0, 0))
        self.label_10.setText("")
        self.label_10.setPixmap(QtGui.QPixmap("./Dataset/train/train_37.png"))
        self.label_10.setScaledContents(False)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 1, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy)
        self.label_13.setBaseSize(QtCore.QSize(0, 0))
        self.label_13.setText("")
        self.label_13.setPixmap(QtGui.QPixmap("./Dataset/train/train_11.png"))
        self.label_13.setScaledContents(False)
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 0, 0, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_17.sizePolicy().hasHeightForWidth())
        self.label_17.setSizePolicy(sizePolicy)
        self.label_17.setBaseSize(QtCore.QSize(0, 0))
        self.label_17.setText("")
        self.label_17.setPixmap(QtGui.QPixmap("./Dataset/train/train_4.png"))
        self.label_17.setScaledContents(False)
        self.label_17.setObjectName("label_17")
        self.gridLayout.addWidget(self.label_17, 0, 4, 1, 1)
        self.label_29 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_29.sizePolicy().hasHeightForWidth())
        self.label_29.setSizePolicy(sizePolicy)
        self.label_29.setBaseSize(QtCore.QSize(0, 0))
        self.label_29.setText("")
        self.label_29.setPixmap(QtGui.QPixmap("./Dataset/train/train_51.png"))
        self.label_29.setScaledContents(False)
        self.label_29.setObjectName("label_29")
        self.gridLayout.addWidget(self.label_29, 4, 2, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_21.sizePolicy().hasHeightForWidth())
        self.label_21.setSizePolicy(sizePolicy)
        self.label_21.setBaseSize(QtCore.QSize(0, 0))
        self.label_21.setText("")
        self.label_21.setPixmap(QtGui.QPixmap("./Dataset/train/train_62.png"))
        self.label_21.setScaledContents(False)
        self.label_21.setObjectName("label_21")
        self.gridLayout.addWidget(self.label_21, 2, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy)
        self.label_14.setBaseSize(QtCore.QSize(0, 0))
        self.label_14.setText("")
        self.label_14.setPixmap(QtGui.QPixmap("./Dataset/train/train_27.png"))
        self.label_14.setScaledContents(False)
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 0, 1, 1, 1)
        self.label_28 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_28.sizePolicy().hasHeightForWidth())
        self.label_28.setSizePolicy(sizePolicy)
        self.label_28.setBaseSize(QtCore.QSize(0, 0))
        self.label_28.setText("")
        self.label_28.setPixmap(QtGui.QPixmap("./Dataset/train/train_169.png"))
        self.label_28.setScaledContents(False)
        self.label_28.setObjectName("label_28")
        self.gridLayout.addWidget(self.label_28, 4, 3, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
        self.label_18.setSizePolicy(sizePolicy)
        self.label_18.setBaseSize(QtCore.QSize(0, 0))
        self.label_18.setText("")
        self.label_18.setPixmap(QtGui.QPixmap("./Dataset/train/train_249.png"))
        self.label_18.setScaledContents(False)
        self.label_18.setObjectName("label_18")
        self.gridLayout.addWidget(self.label_18, 3, 4, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setBaseSize(QtCore.QSize(0, 0))
        self.label_5.setText("")
        self.label_5.setPixmap(QtGui.QPixmap("./Dataset/train/train_5.png"))
        self.label_5.setScaledContents(False)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setBaseSize(QtCore.QSize(0, 0))
        self.label_9.setText("")
        self.label_9.setPixmap(QtGui.QPixmap("./Dataset/train/train_1.png"))
        self.label_9.setScaledContents(False)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 1, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setBaseSize(QtCore.QSize(0, 0))
        self.label_12.setText("")
        self.label_12.setPixmap(QtGui.QPixmap("./Dataset/train/train_6.png"))
        self.label_12.setScaledContents(False)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 1, 4, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy)
        self.label_16.setBaseSize(QtCore.QSize(0, 0))
        self.label_16.setText("")
        self.label_16.setPixmap(QtGui.QPixmap("./Dataset/train/train_36.png"))
        self.label_16.setScaledContents(False)
        self.label_16.setObjectName("label_16")
        self.gridLayout.addWidget(self.label_16, 3, 2, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_22.sizePolicy().hasHeightForWidth())
        self.label_22.setSizePolicy(sizePolicy)
        self.label_22.setBaseSize(QtCore.QSize(0, 0))
        self.label_22.setText("")
        self.label_22.setPixmap(QtGui.QPixmap("./Dataset/train/train_39.png"))
        self.label_22.setScaledContents(False)
        self.label_22.setObjectName("label_22")
        self.gridLayout.addWidget(self.label_22, 2, 1, 1, 1)
        self.label_26 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_26.sizePolicy().hasHeightForWidth())
        self.label_26.setSizePolicy(sizePolicy)
        self.label_26.setBaseSize(QtCore.QSize(0, 0))
        self.label_26.setText("")
        self.label_26.setPixmap(QtGui.QPixmap("./Dataset/train/train_29.png"))
        self.label_26.setScaledContents(False)
        self.label_26.setObjectName("label_26")
        self.gridLayout.addWidget(self.label_26, 2, 2, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setBaseSize(QtCore.QSize(0, 0))
        self.label_11.setText("")
        self.label_11.setPixmap(QtGui.QPixmap("./Dataset/train/train_127.png"))
        self.label_11.setScaledContents(False)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 4, 1, 1, 1)
        self.label_30 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_30.sizePolicy().hasHeightForWidth())
        self.label_30.setSizePolicy(sizePolicy)
        self.label_30.setBaseSize(QtCore.QSize(0, 0))
        self.label_30.setText("")
        self.label_30.setPixmap(QtGui.QPixmap("./Dataset/train/train_43.png"))
        self.label_30.setScaledContents(False)
        self.label_30.setObjectName("label_30")
        self.gridLayout.addWidget(self.label_30, 1, 2, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setBaseSize(QtCore.QSize(0, 0))
        self.label_7.setText("")
        self.label_7.setPixmap(QtGui.QPixmap("./Dataset/train/train_2.png"))
        self.label_7.setScaledContents(False)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 4, 4, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_19.sizePolicy().hasHeightForWidth())
        self.label_19.setSizePolicy(sizePolicy)
        self.label_19.setBaseSize(QtCore.QSize(0, 0))
        self.label_19.setText("")
        self.label_19.setPixmap(QtGui.QPixmap("./Dataset/train/train_177.png"))
        self.label_19.setScaledContents(False)
        self.label_19.setObjectName("label_19")
        self.gridLayout.addWidget(self.label_19, 3, 1, 1, 1)
        self.label_24 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_24.sizePolicy().hasHeightForWidth())
        self.label_24.setSizePolicy(sizePolicy)
        self.label_24.setBaseSize(QtCore.QSize(0, 0))
        self.label_24.setText("")
        self.label_24.setPixmap(QtGui.QPixmap("./Dataset/train/train_69.png"))
        self.label_24.setScaledContents(False)
        self.label_24.setObjectName("label_24")
        self.gridLayout.addWidget(self.label_24, 1, 3, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_20.sizePolicy().hasHeightForWidth())
        self.label_20.setSizePolicy(sizePolicy)
        self.label_20.setBaseSize(QtCore.QSize(0, 0))
        self.label_20.setText("")
        self.label_20.setPixmap(QtGui.QPixmap("./Dataset/train/train_14.png"))
        self.label_20.setScaledContents(False)
        self.label_20.setObjectName("label_20")
        self.gridLayout.addWidget(self.label_20, 3, 0, 1, 1)
        self.label_25 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_25.sizePolicy().hasHeightForWidth())
        self.label_25.setSizePolicy(sizePolicy)
        self.label_25.setBaseSize(QtCore.QSize(0, 0))
        self.label_25.setText("")
        self.label_25.setPixmap(QtGui.QPixmap("./Dataset/train/train_32.png"))
        self.label_25.setScaledContents(False)
        self.label_25.setObjectName("label_25")
        self.gridLayout.addWidget(self.label_25, 0, 3, 1, 1)
        self.label_31 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.label_31.sizePolicy().hasHeightForWidth())
        self.label_31.setSizePolicy(sizePolicy)
        self.label_31.setBaseSize(QtCore.QSize(0, 0))
        self.label_31.setText("")
        self.label_31.setPixmap(QtGui.QPixmap("./Dataset/train/train_23.png"))
        self.label_31.setScaledContents(False)
        self.label_31.setObjectName("label_31")
        self.gridLayout.addWidget(self.label_31, 0, 2, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(410, 260, 131, 151))
        self.label_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_6.setObjectName("label_6")
        self.LearningRate = QtWidgets.QLabel(self.centralwidget)
        self.LearningRate.setGeometry(QtCore.QRect(500, 289, 67, 17))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.LearningRate.setFont(font)
        self.LearningRate.setObjectName("LearningRate")
        self.optimizer = QtWidgets.QLabel(self.centralwidget)
        self.optimizer.setGeometry(QtCore.QRect(480, 315, 67, 17))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.optimizer.setFont(font)
        self.optimizer.setObjectName("optimizer")
        self.NoiseSize = QtWidgets.QLabel(self.centralwidget)
        self.NoiseSize.setGeometry(QtCore.QRect(510, 341, 67, 17))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.NoiseSize.setFont(font)
        self.NoiseSize.setObjectName("NoiseSize")
        self.BatchSize = QtWidgets.QLabel(self.centralwidget)
        self.BatchSize.setGeometry(QtCore.QRect(481, 367, 67, 17))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.BatchSize.setFont(font)
        self.BatchSize.setObjectName("BatchSize")
        self.MaxEpoch = QtWidgets.QLabel(self.centralwidget)
        self.MaxEpoch.setGeometry(QtCore.QRect(482, 393, 67, 17))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.MaxEpoch.setFont(font)
        self.MaxEpoch.setObjectName("MaxEpoch")
        self.Statuslabel = QtWidgets.QLabel(self.centralwidget)
        self.Statuslabel.setGeometry(QtCore.QRect(410, 430, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.Statuslabel.setFont(font)
        self.Statuslabel.setObjectName("Statuslabel")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(155, 320, 140, 131))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_3.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setSpacing(4)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_33 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_33.setEnabled(True)
        self.label_33.setText("")
        self.label_33.setObjectName("label_33")
        self.gridLayout_3.addWidget(self.label_33, 1, 2, 1, 1)
        self.label_32 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_32.setText("")
        self.label_32.setObjectName("label_32")
        self.gridLayout_3.addWidget(self.label_32, 1, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 2, 2, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.gridLayout_3.addWidget(self.label_8, 2, 0, 1, 1)
        self.label_34 = QtWidgets.QLabel(self.centralwidget)
        self.label_34.setGeometry(QtCore.QRect(156, 299, 67, 17))
        self.label_34.setObjectName("label_34")
        self.label_35 = QtWidgets.QLabel(self.centralwidget)
        self.label_35.setGeometry(QtCore.QRect(229, 299, 67, 17))
        self.label_35.setObjectName("label_35")
        self.epochlabel = QtWidgets.QLabel(self.centralwidget)
        self.epochlabel.setGeometry(QtCore.QRect(290, 299, 67, 17))
        self.epochlabel.setObjectName("epochlabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 599, 25))
        self.menubar.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.menubar.setDefaultUp(False)
        self.menubar.setObjectName("menubar")
        self.menuSettings = QtWidgets.QMenu(self.menubar)
        self.menuSettings.setObjectName("menuSettings")
        self.menuLearning_Rate = QtWidgets.QMenu(self.menuSettings)
        self.menuLearning_Rate.setObjectName("menuLearning_Rate")
        self.menuOptimizer = QtWidgets.QMenu(self.menuSettings)
        self.menuOptimizer.setObjectName("menuOptimizer")
        self.menuMax_Epoch = QtWidgets.QMenu(self.menuSettings)
        self.menuMax_Epoch.setObjectName("menuMax_Epoch")
        self.menuBatch_Size = QtWidgets.QMenu(self.menuSettings)
        self.menuBatch_Size.setObjectName("menuBatch_Size")
        self.menuInput_Noise_Size = QtWidgets.QMenu(self.menuSettings)
        self.menuInput_Noise_Size.setObjectName("menuInput_Noise_Size")
        self.menuTraining = QtWidgets.QMenu(self.menubar)
        self.menuTraining.setObjectName("menuTraining")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionStart_Training = QtWidgets.QAction(MainWindow)
        self.actionStart_Training.setStatusTip("")
        self.actionStart_Training.setObjectName("actionStart_Training")
        self.actionStop_Training = QtWidgets.QAction(MainWindow)
        self.actionStop_Training.setObjectName("actionStop_Training")
        self.actionSave_Model = QtWidgets.QAction(MainWindow)
        self.actionSave_Model.setObjectName("actionSave_Model")
        self.actionShow_Test_Result = QtWidgets.QAction(MainWindow)
        self.actionShow_Test_Result.setObjectName("actionShow_Test_Result")
        self.Ada = QtWidgets.QAction(MainWindow)
        self.Ada.setObjectName("Ada")
        self.RMSProp = QtWidgets.QAction(MainWindow)
        self.RMSProp.setObjectName("RMSProp")
        self.Adam = QtWidgets.QAction(MainWindow)
        self.Adam.setObjectName("Adam")
        self.Max50 = QtWidgets.QAction(MainWindow)
        self.Max50.setObjectName("Max50")
        self.Max100 = QtWidgets.QAction(MainWindow)
        self.Max100.setObjectName("Max100")
        self.Max200 = QtWidgets.QAction(MainWindow)
        self.Max200.setObjectName("Max200")
        self.Max500 = QtWidgets.QAction(MainWindow)
        self.Max500.setObjectName("Max500")
        self.Max1000 = QtWidgets.QAction(MainWindow)
        self.Max1000.setObjectName("Max1000")
        self.action2000 = QtWidgets.QAction(MainWindow)
        self.action2000.setObjectName("action2000")
        self.action1e_3 = QtWidgets.QAction(MainWindow)
        self.action1e_3.setObjectName("action1e_3")
        self.action1e_4 = QtWidgets.QAction(MainWindow)
        self.action1e_4.setObjectName("action1e_4")
        self.action1e_5 = QtWidgets.QAction(MainWindow)
        self.action1e_5.setObjectName("action1e_5")
        self.action1e_6 = QtWidgets.QAction(MainWindow)
        self.action1e_6.setObjectName("action1e_6")
        self.Batch50 = QtWidgets.QAction(MainWindow)
        self.Batch50.setObjectName("Batch50")
        self.Batch100 = QtWidgets.QAction(MainWindow)
        self.Batch100.setObjectName("Batch100")
        self.Batch150 = QtWidgets.QAction(MainWindow)
        self.Batch150.setObjectName("Batch150")
        self.Batch200 = QtWidgets.QAction(MainWindow)
        self.Batch200.setObjectName("Batch200")
        self.Batch300 = QtWidgets.QAction(MainWindow)
        self.Batch300.setObjectName("Batch300")
        self.action500_3 = QtWidgets.QAction(MainWindow)
        self.action500_3.setObjectName("action500_3")
        self.Noise50 = QtWidgets.QAction(MainWindow)
        self.Noise50.setObjectName("Noise50")
        self.Noise100 = QtWidgets.QAction(MainWindow)
        self.Noise100.setObjectName("Noise100")
        self.Noise150 = QtWidgets.QAction(MainWindow)
        self.Noise150.setObjectName("Noise150")
        self.Noise200 = QtWidgets.QAction(MainWindow)
        self.Noise200.setObjectName("Noise200")
        self.action250 = QtWidgets.QAction(MainWindow)
        self.action250.setObjectName("action250")
        self.Noise300 = QtWidgets.QAction(MainWindow)
        self.Noise300.setObjectName("Noise300")
        self.menuLearning_Rate.addAction(self.action1e_3)
        self.menuLearning_Rate.addAction(self.action1e_4)
        self.menuLearning_Rate.addAction(self.action1e_5)
        self.menuLearning_Rate.addAction(self.action1e_6)
        self.menuOptimizer.addAction(self.Ada)
        self.menuOptimizer.addAction(self.RMSProp)
        self.menuOptimizer.addAction(self.Adam)
        self.menuMax_Epoch.addAction(self.Max50)
        self.menuMax_Epoch.addAction(self.Max100)
        self.menuMax_Epoch.addAction(self.Max200)
        self.menuMax_Epoch.addAction(self.Max500)
        self.menuMax_Epoch.addAction(self.Max1000)
        self.menuBatch_Size.addAction(self.Batch50)
        self.menuBatch_Size.addAction(self.Batch100)
        self.menuBatch_Size.addAction(self.Batch150)
        self.menuBatch_Size.addAction(self.Batch200)
        self.menuBatch_Size.addAction(self.Batch300)
        self.menuInput_Noise_Size.addAction(self.Noise50)
        self.menuInput_Noise_Size.addAction(self.Noise100)
        self.menuInput_Noise_Size.addAction(self.Noise150)
        self.menuInput_Noise_Size.addAction(self.Noise200)
        self.menuInput_Noise_Size.addAction(self.Noise300)
        self.menuSettings.addAction(self.menuLearning_Rate.menuAction())
        self.menuSettings.addAction(self.menuOptimizer.menuAction())
        self.menuSettings.addAction(self.menuBatch_Size.menuAction())
        self.menuSettings.addAction(self.menuInput_Noise_Size.menuAction())
        self.menuSettings.addAction(self.menuMax_Epoch.menuAction())
        self.menuTraining.addAction(self.actionStart_Training)
        self.menuTraining.addAction(self.actionStop_Training)
        self.menubar.addAction(self.menuTraining.menuAction())
        self.menubar.addAction(self.menuSettings.menuAction())

        self.retranslateUi(MainWindow)
        self.action1e_3.triggered.connect(self.lr_1e_3)
        self.action1e_4.triggered.connect(self.lr_1e_4)
        self.action1e_5.triggered.connect(self.lr_1e_5)
        self.action1e_6.triggered.connect(self.lr_1e_6)
        self.Noise50.triggered.connect(self.noise_50)
        self.Noise100.triggered.connect(self.noise_100)
        self.Noise150.triggered.connect(self.noise_150)
        self.Noise200.triggered.connect(self.noise_200)
        self.Noise300.triggered.connect(self.noise_300)
        self.Adam.triggered.connect(self.setAdam)
        self.Ada.triggered.connect(self.setAda)
        self.RMSProp.triggered.connect(self.setRMS)
        self.Max50.triggered.connect(self.setMax50)
        self.Max100.triggered.connect(self.setMax100)
        self.Max200.triggered.connect(self.setMax200)
        self.Max500.triggered.connect(self.setMax500)
        self.Max1000.triggered.connect(self.setMax1000)
        self.Batch50.triggered.connect(self.setBatch50)
        self.Batch100.triggered.connect(self.setBatch100)
        self.Batch150.triggered.connect(self.setBatch150)
        self.Batch200.triggered.connect(self.setBatch200)
        self.Batch300.triggered.connect(self.setBatch300)
        self.setdefaultbutton.clicked.connect(self.setalldefault)
        self.actionStart_Training.triggered.connect(self.settraining)
        self.actionStop_Training.triggered.connect(self.setstop)
        self.pausebutton.clicked.connect(self.setpauseresume)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    '''
    settings
    '''
    def settraining(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.Statuslabel.setText('Status: Training...')
            global nettraininglabel
            nettraininglabel = True
            self.th = TrainingThread()
            self.th.start()





    def setstop(self):
        self.Statuslabel.setText('Status: Stop')
        global nettraininglabel
        nettraininglabel = False
        global netpauselabel
        netpauselabel = False

    def setpauseresume(self):
        global netpauselabel
        if self.Statuslabel.text() == 'Status: Pause':
            self.Statuslabel.setText('Status: Training...')
            netpauselabel = False

        elif self.Statuslabel.text() == 'Status: Training...':
            self.Statuslabel.setText('Status: Pause')
            netpauselabel = True

    def setalldefault(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.LearningRate.setText('1e-4')
            global netLR
            netLR = 1e-4
            self.optimizer.setText('Adam')
            global netoptimizerlabel
            netoptimizerlabel = 'Adam'
            self.NoiseSize.setText('100')
            global netInput_Size
            netInput_Size = 100
            self.BatchSize.setText('100')
            global netBatchSize
            netBatchSize = 100
            self.MaxEpoch.setText('100')
            global netMaxEpoch
            netMaxEpoch = 100

    def lr_1e_3(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.LearningRate.setText('1e-3')
            global netLR
            netLR = 1e-3

    def lr_1e_4(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.LearningRate.setText('1e-4')
            global netLR
            netLR = 1e-4

    def lr_1e_5(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.LearningRate.setText('1e-5')
            global netLR
            netLR = 1e-5

    def lr_1e_6(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.LearningRate.setText('1e-6')
            global netLR
            netLR = 1e-6

    def noise_50(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.NoiseSize.setText('50')
            global netInput_Size
            netInput_Size = 50

    def noise_100(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.NoiseSize.setText('100')
            global netInput_Size
            netInput_Size = 100

    def noise_150(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.NoiseSize.setText('150')
            global netInput_Size
            netInput_Size = 150

    def noise_200(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.NoiseSize.setText('200')
            global netInput_Size
            netInput_Size = 200

    def noise_300(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.NoiseSize.setText('300')
            global netInput_Size
            netInput_Size = 300

    def setAda(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.optimizer.setText('Adagrad')
            global netoptimizerlabel
            netoptimizerlabel = 'Adagrad'

    def setAdam(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.optimizer.setText('Adam')
            global netoptimizerlabel
            netoptimizerlabel = 'Adam'

    def setRMS(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.optimizer.setText('RMSProp')
            global netoptimizerlabel
            netoptimizerlabel = 'RMSProp'

    def setMax50(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.MaxEpoch.setText('50')
            global netMaxEpoch
            netMaxEpoch = 50

    def setMax100(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.MaxEpoch.setText('100')
            global netMaxEpoch
            netMaxEpoch = 100

    def setMax200(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.MaxEpoch.setText('200')
            global netMaxEpoch
            netMaxEpoch = 200

    def setMax500(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.MaxEpoch.setText('500')
            global netMaxEpoch
            netMaxEpoch = 500

    def setMax1000(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.MaxEpoch.setText('1000')
            global netMaxEpoch
            netMaxEpoch = 1000

    def setBatch50(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.BatchSize.setText('50')
            global netBatchSize
            netBatchSize = 50

    def setBatch100(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.BatchSize.setText('100')
            global netBatchSize
            netBatchSize = 100

    def setBatch150(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.BatchSize.setText('150')
            global netBatchSize
            netBatchSize = 150

    def setBatch200(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.BatchSize.setText('200')
            global netBatchSize
            netBatchSize = 200

    def setBatch300(self):
        if self.Statuslabel.text() == 'Status: Stop':
            self.BatchSize.setText('300')
            global netBatchSize
            netBatchSize = 300

    def updateepoch(self):
        if self.Statuslabel.text() == 'Status: Training...':
            self.epochlabel.setText(str(now_epoch))
            if (os.path.exists("./Result/show1") and os.path.exists("./Result/show2") and os.path.exists('./Result/show3') and os.path.exists('./Result/show4'))or os.path.exists('./Dataset/train/train_98.png'):
                self.label_32.setPixmap(QtGui.QPixmap("./Result/show1"))
                self.label_33.setPixmap(QtGui.QPixmap("./Result/show2"))
                self.label_8.setPixmap(QtGui.QPixmap("./Result/show3"))
                self.label_2.setPixmap(QtGui.QPixmap("./Result/show4"))
        elif self.Statuslabel.text() == 'Status: Stop':
            self.epochlabel.setText(str(now_epoch))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Mnist GAN"))
        self.pausebutton.setText(_translate("MainWindow", "Pause/Resume"))
        self.setdefaultbutton.setText(_translate("MainWindow", "Set to Default"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p>Welcome to Mnist GAN Training Program</p><p align=\"right\"><span style=\" font-size:8pt;\">made by Vince (Yicheng Wang)</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">Default Settings:</span><span style=\" font-size:16pt;\"/></p><p><span style=\" font-size:9pt;\">Learning Rate = 1e-4 </span></p><p><span style=\" font-size:9pt;\">Optimizer = Adam </span></p><p><span style=\" font-size:9pt;\">Input Noise Size = 100 </span></p><p><span style=\" font-size:9pt;\">Batch Size = 100</span></p><p><span style=\" font-size:9pt;\">Max Epoch = 100</span></p><p><br/></p></body></html>"))
        self.label.setText(_translate("MainWindow", "Mnist Dataset"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">Recent Settings:</span></p><p><span style=\" font-size:9pt;\">Learning Rate = </span></p><p><span style=\" font-size:9pt;\">Optimizer = </span></p><p><span style=\" font-size:9pt;\">Input Noise Size = </span></p><p><span style=\" font-size:9pt;\">Batch Size = </span></p><p><span style=\" font-size:9pt;\">Max Epoch = </span></p><p><br/></p></body></html>"))
        self.LearningRate.setText(_translate("MainWindow", "1e-4"))
        self.optimizer.setText(_translate("MainWindow", "Adam"))
        self.NoiseSize.setText(_translate("MainWindow", "100"))
        self.BatchSize.setText(_translate("MainWindow", "100"))
        self.MaxEpoch.setText(_translate("MainWindow", "100"))
        self.Statuslabel.setText(_translate("MainWindow", "Status: Stop"))
        self.label_34.setText(_translate("MainWindow", "Results"))
        self.label_35.setText(_translate("MainWindow", "Epoch = "))
        self.epochlabel.setText(_translate("MainWindow", "1"))
        self.menuSettings.setTitle(_translate("MainWindow", "Settings"))
        self.menuLearning_Rate.setTitle(_translate("MainWindow", "Learning Rate"))
        self.menuOptimizer.setTitle(_translate("MainWindow", "Optimizer"))
        self.menuMax_Epoch.setTitle(_translate("MainWindow", "Max Epoch"))
        self.menuBatch_Size.setTitle(_translate("MainWindow", "Batch Size"))
        self.menuInput_Noise_Size.setTitle(_translate("MainWindow", "Input Noise Size"))
        self.menuTraining.setTitle(_translate("MainWindow", "Train"))
        self.actionStart_Training.setText(_translate("MainWindow", "Start Training"))
        self.actionStop_Training.setText(_translate("MainWindow", "Stop Training"))
        self.actionSave_Model.setText(_translate("MainWindow", "Save Model"))
        self.actionShow_Test_Result.setText(_translate("MainWindow", "Show Test Result"))
        self.Ada.setText(_translate("MainWindow", "Adagrad"))
        self.RMSProp.setText(_translate("MainWindow", "RMSprop"))
        self.Adam.setText(_translate("MainWindow", "Adam"))
        self.Max50.setText(_translate("MainWindow", "50"))
        self.Max100.setText(_translate("MainWindow", "100"))
        self.Max200.setText(_translate("MainWindow", "200"))
        self.Max500.setText(_translate("MainWindow", "500"))
        self.Max1000.setText(_translate("MainWindow", "1000"))
        self.action2000.setText(_translate("MainWindow", "2000"))
        self.action1e_3.setText(_translate("MainWindow", "1e-3"))
        self.action1e_4.setText(_translate("MainWindow", "1e-4"))
        self.action1e_5.setText(_translate("MainWindow", "1e-5"))
        self.action1e_6.setText(_translate("MainWindow", "1e-6"))
        self.Batch50.setText(_translate("MainWindow", "50"))
        self.Batch100.setText(_translate("MainWindow", "100"))
        self.Batch150.setText(_translate("MainWindow", "150"))
        self.Batch200.setText(_translate("MainWindow", "200"))
        self.Batch300.setText(_translate("MainWindow", "300"))
        self.action500_3.setText(_translate("MainWindow", "500"))
        self.Noise50.setText(_translate("MainWindow", "50"))
        self.Noise100.setText(_translate("MainWindow", "100"))
        self.Noise150.setText(_translate("MainWindow", "150"))
        self.Noise200.setText(_translate("MainWindow", "200"))
        self.action250.setText(_translate("MainWindow", "250"))
        self.Noise300.setText(_translate("MainWindow", "300"))

if __name__ == '__main__':
  import sys
  app = QtWidgets.QApplication(sys.argv)
  MainWindow = QtWidgets.QMainWindow()
  ui = Ui_MainWindow()
  ui.setupUi(MainWindow)
  update_data_thread = Updateresult()
  update_data_thread.trigger.connect(ui.updateepoch)
  update_data_thread.start()
  MainWindow.show()
  sys.exit(app.exec_())






