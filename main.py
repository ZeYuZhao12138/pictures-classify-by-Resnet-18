# 2022/08/19 16:51
# have a good day!
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import ToTensor

from torch.nn import functional as F
import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self, input_channel, output_channel, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel,
                              out_channels=output_channel,
                              kernel_size=3,
                              padding=1,
                              stride=strides)
        self.conv2 = nn.Conv2d(in_channels=output_channel,
                               out_channels=output_channel,
                               kernel_size=3,
                               padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channel,
                                   out_channels=output_channel,
                                   kernel_size=1,
                                   stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)


def resnet_block(input_channels, output_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channel=input_channels,
                         output_channel=output_channels,
                         use_1x1conv=True,
                         strides=2)
            )
        else:
            blk.append(
                Residual(input_channel=output_channels,
                         output_channel=output_channels)
            )
    return blk


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.image_path = self.root_dir + self.label_dir
        self.img_list = os.listdir(self.image_path)
        self.tensor_trans = ToTensor()

    def __getitem__(self, item):
        img = Image.open(self.root_dir + self.label_dir + '/' + self.img_list[item])
        img_tensor = self.tensor_trans(img)
        if self.label_dir == 'cats':
            label = 0
        else:
            label = 1
        return img_tensor, label

    def __len__(self):
        return len(self.img_list)


# def my_collate(batch):
#     img = [each[0] for each in batch]
#     label = [each[1] for each in batch]
#     return img, label

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader)
    for batch_num, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_num % 2 == 0:
            loss, current = loss.item(), batch_num
            print(f"loss: {loss}, current: {current}/{size}")


def test(dataloader, model):
    model.eval()
    loss, correct = 0, 0
    for t, (X, y) in enumerate(dataloader):
        pred = F.softmax(model(X))
        if pred.argmax(1) == y:
            correct += 1
        print(f"Accuracy:{correct / (t+1)}")



train_data_path = 'data/train/'
test_data_path = 'data/test/'
dog_label_path = 'dogs'
cat_label_path = 'cats'


cats_dataset = MyData(train_data_path, cat_label_path)
dogs_dataset = MyData(train_data_path, dog_label_path)
cats_testdata = MyData(test_data_path, cat_label_path)
dogs_testdata = MyData(test_data_path, dog_label_path)


dataset = cats_dataset + dogs_dataset
testdataset = cats_testdata + dogs_testdata

BATCH_SIZE = 1

dataloader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        # collate_fn=my_collate
                        )
test_dataloader = DataLoader(testdataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        # collate_fn=my_collate
                        )

b1 = nn.Sequential(nn.Conv2d(in_channels=3,
                             out_channels=64,
                             kernel_size=7,
                             stride=2,
                             padding=1),
                   nn.BatchNorm2d(64),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3,
                                stride=2,
                                padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(512, 2))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

epochs = 1
for t in range(epochs):
    print(f"epoch:{t}\n")
    train(dataloader, net, loss_fn, optimizer)
print('done')

test(test_dataloader, net)