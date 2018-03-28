import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, ReLU, Softmax, MaxPool2d, AvgPool2d, ConvTranspose2d, Upsample, Sigmoid
from torch.nn import BatchNorm2d as Norm2d


class Net(nn.Module):
    def __init__(self, nclass, conv_ncs=[1]):
        super(Net, self).__init__()
        
        conv_blocks = []
        for nlayer in range(len(conv_ncs)):
            if nlayer is 0:
                conv_blocks.append(conv357_block(1, conv_ncs[nlayer]))
            else:
                conv_blocks.append(conv357_block(3*conv_ncs[nlayer-1], conv_ncs[nlayer]))
        self.conv_blocks = conv_blocks
        self.dense = Linear(int(3*conv_ncs[-1]*224./pow(2, len(conv_ncs))), nclass)
        
    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            x = MaxPool(2)(x)
        output = Softmax()(self.dense(x.view(-1)))
        
        return output

class conv357_block(nn.Module):
    def __init__(self, nc, nf, norm_flag=True):
        super(conv357_block, self).__init__()
        
        self.conv3 = Conv2d(nc, nf, 3, 1, 1, bias=True)
        self.conv5 = Conv2d(nc, nf, 5, 1, 2, bias=True)
        self.conv7 = Conv2d(nc, nf, 7, 1, 3, bias=True)
        
        if norm_flag:
            self.norm_flag = norm_flag
            self.norm = Norm2d(3*nf)
        
        self.weights_init()
        
    def forward(self, x):
        # x size BCHW
        conv3 = ReLU(True)(self.conv3(x))
        conv5 = ReLU(True)(self.conv5(x))
        conv7 = ReLU(True)(self.conv7(x))
        
        cat = torch.cat((conv3, conv5, conv7), dim=1)
        if self.norm_flag: cat = self.norm(cat)
        
        return cat

    def weights_init(self, init_func=torch.nn.init.kaiming_uniform):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight)


class Encoder(nn.Module):
    def __init__(self, nc, img_h, img_w, ndf=16):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 224 x 224
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 112 x 112
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 56 x 56
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 28 x 28
            nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 14 x 14
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            # nn.BatchNorm2d(ndf * 4),
            # state size. (ndf*4) x 7 x 7
        )
        
    def weights_init(self, init_func=torch.nn.init.kaiming_normal):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_func(m)

    def forward(self, x):
        output = self.main(x)
        return output


class Decoder(nn.Module):
    def __init__(self, nc, img_h, img_w, ngf=16):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            # input is (ndf*4) x 7 x 7, going into a convolution
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 14 x 14
            nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 28 x 28
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 56 x 56
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 112 x 112
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 224 x 224
        )

    def weights_init(self, init_func=torch.nn.init.kaiming_normal):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_func(m)
        
    def forward(self, x):
        output = self.main(x)
        return output


class VAE(nn.Module):
    def __init__(self, nc, img_h, img_w):
        super(VAE, self).__init__()
        self.encoder1 = Encoder(nc, img_h, img_w)
        self.encoder2 = Encoder(nc, img_h, img_w)
        self.decoder = Decoder(nc, img_h, img_w)
        self.encoder1.weights_init()
        self.encoder2.weights_init()
        self.decoder.weights_init()
    
    def reparameterize(self, mu, logvar):
        if self.training:
            eps = Variable(std.data.new(std.size()).normal_())
            std = logvar.mul(0.5).exp_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, x):
        mu, logvar = self.encoder1(x), self.encoder2(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

''' VAE Loss Function

def loss_function(recon_x, x, mu, logvar):
    reconstruction_function = nn.BCELoss(size_average=False)
    BCE = reconstruction_function(recon_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD
'''