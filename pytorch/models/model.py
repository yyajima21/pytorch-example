import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):

    def __init__(self, im_size, kernel_size):

        super(UNet, self).__init__()
        self.conv1_1 = nn.Conv2d(im_size[0], 32, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size, padding=(kernel_size - 1) // 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size, padding=(kernel_size - 1) // 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size, padding=(kernel_size - 1) // 2)
        self.transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(128, 64, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size, padding=(kernel_size - 1) // 2)
        self.transpose2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(64, 32, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv5_2 = nn.Conv2d(32, 32, kernel_size, padding=(kernel_size - 1) // 2)
        self.predict = nn.Conv2d(32, 1, kernel_size=1)
                

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
        produce a score for each class.

        Arguments:
        images (Variable): A tensor of size (N, C, H, W) where
            N is the batch size
            C is the number of channels
            H is the image height
            W is the image width

        Returns:
        A torch Variable of size (N, n_classes) specifying the score
        for each example and category.
        '''
        scores = None

        down1 = F.relu(self.conv1_1(images))
        down1 = F.relu(self.conv1_2(down1))
        down2 = self.pool1(down1)

        down2 = F.relu(self.conv2_1(down2))
        down2 = F.relu(self.conv2_2(down2))
        down3 = self.pool2(down2)

        up1 = F.relu(self.conv3_1(down3))
        up1 = F.relu(self.conv3_2(up1))
        up2 = self.transpose1(up1)

        up2 = torch.cat([down2, up2], dim=1)
        up2 = F.relu(self.conv4_1(up2))
        up2 = F.relu(self.conv4_2(up2))
        up3 = self.transpose2(up2)

        up3 = torch.cat([down1, up3], dim=1)
        up3 = F.relu(self.conv5_1(up3))
        up3 = F.relu(self.conv5_2(up3))
        scores = torch.sigmoid(self.predict(up3))

        return scores 

class NeuroModel(nn.Module):

    def __init__(self, im_size, kernel_size):

        super(NeuroModel, self).__init__()
        self.conv1_1 = nn.Conv2d(im_size[0], 32, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv1_3 = nn.Conv2d(32, 64, kernel_size, padding=(kernel_size - 1) // 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 64, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2_3 = nn.Conv2d(64, 128, kernel_size, padding=(kernel_size - 1) // 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 128, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv3_3 = nn.Conv2d(128, 256, kernel_size, padding=(kernel_size - 1) // 2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 256, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv4_3 = nn.Conv2d(256, 256, kernel_size, padding=(kernel_size - 1) // 2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(im_size[1]/16 * im_size[2]/16 * 256, 250)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(250, 250)
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(250, 4)

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
        produce a score for each class.

        Arguments:
        images (Variable): A tensor of size (N, C, H, W) where
            N is the batch size
            C is the number of channels
            H is the image height
            W is the image width

        Returns:
        A torch Variable of size (N, n_classes) specifying the score
        for each example and category.
        '''
        scores = None

        net = F.relu(self.conv1_1(images))
        net = F.relu(self.conv1_2(net))
        net = F.relu(self.conv1_3(net))
        net = self.pool1(net)

        net = F.relu(self.conv2_1(net))
        net = F.relu(self.conv2_2(net))
        net = F.relu(self.conv2_3(net))
        net = self.pool2(net)

        net = F.relu(self.conv3_1(net))
        net = F.relu(self.conv3_2(net))
        net = F.relu(self.conv3_3(net))
        net = self.pool3(net)

        net = F.relu(self.conv4_1(net))
        net = F.relu(self.conv4_2(net))
        net = F.relu(self.conv4_3(net))
        net = self.pool4(net)

        N, C, H, W = net.shape
        reshaped = net.view(N, C*H*W)
        scores = self.fc3(self.dropout2(self.fc2(self.dropout1(self.fc1(reshaped)))))

        return scores
