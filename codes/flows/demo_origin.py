import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cv2
import numpy as np
from skimage import segmentation

"""
Unsupervised Segmentation 
This pytorch code generates segmentation labels of an input image.

![Unsupervised Segmentation](https://github.com/kanezaki/pytorch-unsupervised-segmentation/blob/gh-pages/ICASSP2018_kanezaki.png)

Asako Kanezaki.
**Unsupervised Image Segmentation by Backpropagation.** 
*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2018.
([pdf](https://kanezaki.github.io/pytorch-unsupervised-segmentation/ICASSP2018_kanezaki.pdf))
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)  # choose GPU:0

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=100, type=int,
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=128, type=int,
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int,
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int,
                    help='number of convolutional layers')
parser.add_argument('--num_superpixels', metavar='K', default=10000, type=int,
                    help='number of superpixels')
parser.add_argument('--compactness', metavar='C', default=100, type=float,
                    help='compactness of superpixels')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME', default='image/woof.jpg', type=str,
                    help='input image file name', )
args = parser.parse_args()


# CNN model
class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = []
        self.bn2 = []
        for i in range(args.nConv - 1):
            self.conv2.append(nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(args.nChannel))
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(args.nConv - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


# load image
im = cv2.imread(args.input)
data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
if use_cuda:
    data = data.cuda()
data = Variable(data)

# slic
labels = segmentation.slic(im, compactness=args.compactness, n_segments=args.num_superpixels)
# labels = segmentation.slic(im, compactness=args.compactness, n_segments=args.num_superpixels, max_iter=1)
labels = labels.reshape(im.shape[0] * im.shape[1])
u_labels = np.unique(labels)
l_inds = []
for i in range(len(u_labels)):
    l_inds.append(np.where(labels == u_labels[i])[0])

# train
model = MyNet(data.size(1))
if use_cuda:
    model.cuda()
    for i in range(args.nConv - 1):
        model.conv2[i].cuda()
        model.bn2[i].cuda()
model.train()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255, size=(100, 3))

from time import time
start_time = time()

for batch_idx in range(args.maxIter):
    # forwarding
    optimizer.zero_grad()
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))
    if args.visualize:
        im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
        cv2.imshow("output", im_target_rgb)
        cv2.waitKey(1)
        # if np.log2(batch_idx) % 1 == 0:
        #     cv2.imwrite("output_%s_%02i.jpg" % (args.input, batch_idx), im_target_rgb)

    # superpixel refinement
    # TODO: use Torch Variable instead of numpy for faster calculation
    for i in range(len(l_inds)):
        labels_per_sp = im_target[l_inds[i]]
        u_labels_per_sp = np.unique(labels_per_sp)
        hist = np.zeros(len(u_labels_per_sp))
        for j in range(len(hist)):
            hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
        im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
    target = torch.from_numpy(im_target)
    if use_cuda:
        target = target.cuda()
    target = Variable(target)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    print(batch_idx, '/', args.maxIter, ':', nLabels, loss.data[0])
    # if nLabels <= args.minLabels:
    #     print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
    #     break
time1= int(time() - start_time)
print('TimeUsed: %.2f' % time1)

# save output image
if not args.visualize:
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
cv2.imwrite("output_%s_128_%is.jpg" % (args.input[6:-4], time1), im_target_rgb)

