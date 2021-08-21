# -*- coding: utf-8 -*-

#   [1] General code structure based on https://gist.github.com/cwkx/e63ea58890a496d65467761a879c717a
#       by Willcocks, Chris

#   [2] Basic understanding of GAN implementation from:
#       https://gist.github.com/cwkx/74e33bc96f94f381bd15032d57e43786#file-simple-gan-ipynb
#       by Willcocks, Chris

#   [3] DCGAN structure based on example found in Pytorch tutorials at:
#       https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

# run in notebooks to see full features

from __future__ import print_function
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# percentage of plane images used
plane_prob = 0.0
# percentage of bird images used
bird_prob = 0.0
batch_size = 64
num_epochs = 160
image_size = 64
# number of channels
nc = 3
# length of latent vector z
nz = 100
# number of generator features
ngf = 64
# number of discriminator features
ndf = 64
# learning rate for Adam optimiser
lr = 0.0002
# beta parameter for Adam
beta1 = 0.5

# transform pipeline to normalise image data
my_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# filters input dataset according to horse, bird and airplane labels,
# percentage of each image type included controlled by parameters
def get_percent_indexes(target, plane_prob, bird_prob):
    label_indices = []
    for i in range(len(target)):
        if target[i] == 7:
            label_indices.append(i)
        elif (target[i] == 0) and (random.uniform(0, 1) < plane_prob):
            label_indices.append(i)
        elif (target[i] == 2) and (random.uniform(0, 1) < bird_prob):
            label_indices.append(i)
    return label_indices


dataset = dset.CIFAR10('drive/My Drive/training/cifar10', train=True, download=True, transform=my_transforms)
# classes of images
classes = [i[1] for i in dataset]
train_indices = get_percent_indexes(classes, plane_prob, bird_prob)

select_set = torch.utils.data.Subset(dataset, train_indices)

dataloader = torch.utils.data.DataLoader(dataset=select_set, shuffle=True,
                                         batch_size=batch_size)
# use cuda if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# weight initialisation, generator and discriminator based on [3]
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# added Dropouts to Generator to introduce noise as discussed in paper
# replacing ReLU with Leaky variant was experimented with however
# yielded worse results and rapid model collapse
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.Dropout(),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.Dropout(),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Dropouts were experimented with in the discriminator however they made more sense in the generator
# as they forced it to work to restore the original image not just fool D
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Used for testing discriminator on inputs
def trial(D, label, criterion, test_img, size):
    label_tensor = torch.full((size,), label, dtype=torch.float, device=device)
    judgement = D(test_img).view(-1)
    err = criterion(judgement, label_tensor)
    return err, judgement


# Create Generator and apply preset weights to conv and batch norms
Gen = Generator().to(device)
Gen.apply(weights_init)

Dis = Discriminator().to(device)
Dis.apply(weights_init)
# BCE Loss used as in [1] in-line with research findings cited in paper
criterion = nn.BCELoss()
# fixed noise is used to judge the changing performance of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1.
fake_label = 0.

# The below optimiser was experimented with as shown in the literature however yielded worse results.
# optimizerD = optim.SGD(netD.parameters(), lr=lr, momentum=0.5)
Adam_D = optim.Adam(Dis.parameters(), lr=lr, betas=(beta1, 0.999))
Adam_G = optim.Adam(Gen.parameters(), lr=lr, betas=(beta1, 0.999))
# follows execution method of [1]
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        Dis.zero_grad()
        real_images = data[0].to(device)
        b_size = real_images.size(0)

        real_d_error, judgement = trial(Dis, real_label, criterion, real_images, b_size)
        real_d_error.backward()
        D_x = judgement.mean().item()
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_image = Gen(noise)
        fake_d_error, judgement = trial(Dis, fake_label, criterion, fake_image.detach(), b_size)
        fake_d_error.backward()
        D_G_z1 = judgement.mean().item()
        errD = real_d_error + fake_d_error
        Adam_D.step()

        Gen.zero_grad()
        errG, judgement = trial(Dis, real_label, criterion, fake_image, b_size)
        errG.backward()
        D_G_z2 = judgement.mean().item()
        Adam_G.step()

        # progress tracking as recommended in [2]
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake_image = Gen(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_image, padding=2, normalize=True))

        iters += 1


# image generation process depicted as in [3]
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())
