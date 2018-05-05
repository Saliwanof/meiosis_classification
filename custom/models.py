import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, Softmax, MaxPool2d, AvgPool2d, ConvTranspose2d, Upsample, Sigmoid
from torch.nn import Dropout, Dropout2d, ReLU, Softplus, LeakyReLU
from torch.nn import BatchNorm2d as Norm2d
from torch.nn import BatchNorm1d as Norm1d

activation_func = ReLU(True)
# activation_func = LeakyReLU(0.2, True)
# activation_func = Softplus()


class Net_Period_Classification(nn.Module):
    '''
    receptfield of size 20px - 40px needed to detect an anomaly
    '''
    def __init__(self, nclass):
        super(Net_Period_Classification, self).__init__()
        
        self.conv1a = conv_bn(1, 32, 3, 1, 2, 2, True)
        self.drop1a = Dropout2d(.2, True)
        self.conv1b = conv_bn(32, 32, 3, 1, 2, 2, True)
        self.drop1b = Dropout2d(.2, True)
        # pool
        # (32, 112, 112)
        self.conv2 = inception_v2(32, 64)
        self.drop2 = Dropout2d(.4, True)
        # pool
        # (192, 56, 56)
        self.conv3 = inception_v2(192, 64)
        self.drop3 = Dropout2d(.4, True)
        # pool
        # (192, 28, 28)
        self.conv4 = inception_v2(192, 64)
        self.drop4 = Dropout2d(.4, True)
        # pool
        # (192, 14, 14)
        self.conv5 = Conv2d(192, 16, 1, 1, 0, 1)
        self.drop5 = Dropout2d(.4, True)
        # (16, 14, 14)
        # pool
        # (16, 7, 7)
        self.mlp1 = Linear(16*7*7, 64)
        self.drop01 = Dropout(.2, True)
        self.mlp2 = Linear(64, 64)
        self.drop02 = Dropout(.2, True)
        self.mlp3 = Linear(64, nclass)
        
    def forward(self, x):
        x = self.conv1a(x)
        x = self.drop1a(x)
        x = self.conv1b(x)
        x = self.drop1b(x)
        x = MaxPool2d(2)(x)
        
        x = self.conv2(x)
        x = self.drop2(x)
        x = MaxPool2d(2)(x)
        
        x = self.conv3(x)
        x = self.drop3(x)
        x = MaxPool2d(2)(x)
        
        x = self.conv4(x)
        x = self.drop4(x)
        x = MaxPool2d(2)(x)
        
        x = self.conv5(x)
        x = self.drop5(x)
        x = MaxPool2d(2)(x)
        
        feature = x
        x = x.view(-1, 16*7*7)
        
        x = activation_func(self.mlp1(x))
        x = self.drop01(x)
        x = activation_func(self.mlp2(x))
        x = self.drop02(x)
        x = self.mlp3(x)
        target = x
        
        return target, feature

class inception_v2(nn.Module):
    def __init__(self, nc, nf, norm_flag=True):
        super(inception_v2, self).__init__()
        
        self.conv3a = Conv2d(nc, nf, 1, 1, 0)
        self.conv3b = Conv2d(nf, nf, 3, 1, 1)
        self.conv5a = Conv2d(nc, nf, 1, 1, 0)
        self.conv5b = Conv2d(nf, nf, 3, 1, 1)
        self.conv5c = Conv2d(nf, nf, 3, 1, 1)
        self.conv7a = Conv2d(nc, nf, 1, 1, 0)
        self.conv7b = Conv2d(nf, nf, 3, 1, 1)
        self.conv7c = Conv2d(nf, nf, 3, 1, 1)
        self.conv7d = Conv2d(nf, nf, 3, 1, 1)
        
        if norm_flag:
            self.norm_flag = norm_flag
            self.norm = Norm2d(3*nf)
        
    def forward(self, x):
        # x size BCHW
        conv3 = activation_func(self.conv3a(x))
        conv3 = activation_func(self.conv3b(conv3))
        conv5 = activation_func(self.conv5a(x))
        conv5 = activation_func(self.conv5b(conv5))
        conv5 = activation_func(self.conv5c(conv5))
        conv7 = activation_func(self.conv7a(x))
        conv7 = activation_func(self.conv7b(conv7))
        conv7 = activation_func(self.conv7c(conv7))
        conv7 = activation_func(self.conv7d(conv7))
        
        cat = torch.cat((conv3, conv5, conv7), dim=1)
        if self.norm_flag: cat = self.norm(cat)
        
        return cat


class inception_v3(nn.Module):
    def __init__(self, nc, nf, norm_flag=True):
        super(inception_v3, self).__init__()
        
        self.conv3a = Conv2d(nc, nf, 1, 1, 0)
        self.conv3b = factorize_conv(nf, nf, 3)
        self.conv5a = Conv2d(nc, nf, 1, 1, 0)
        self.conv5b = factorize_conv(nf, nf, 5)
        self.conv7a = Conv2d(nc, nf, 1, 1, 0)
        self.conv7b = factorize_conv(nf, nf, 7)
        
        if norm_flag:
            self.norm_flag = norm_flag
            self.norm = Norm2d(3*nf)
        
    def forward(self, x):
        # x size BCHW
        conv3 = activation_func(self.conv3a(x))
        conv3 = activation_func(self.conv3b(conv3))
        conv5 = activation_func(self.conv5a(x))
        conv5 = activation_func(self.conv5b(conv5))
        conv7 = activation_func(self.conv7a(x))
        conv7 = activation_func(self.conv7b(conv7))
        
        cat = torch.cat((conv3, conv5, conv7), dim=1)
        if self.norm_flag: cat = self.norm(cat)
        
        return cat

class factorize_conv(nn.Module):
    def __init__(self, nc, nf, ksize):
        super(factorize_conv, self).__init__()
        
        padding = ksize // 2
        self.conv_1xn = Conv2d(nc, nf, (ksize, 1), 1, (padding, 0))
        self.conv_nx1 = Conv2d(nf, nf, (1, ksize), 1, (0, padding))
        
    def forward(self, x):
        x = self.conv_1xn(x)
        x = self.conv_nx1(x)
        
        return x
    
class conv_bn(nn.Module):
    def __init__(self, nc, nf, filter_size=3, stride=1, padding=1, dilation=1, bias_flag=True):
        super(conv_bn, self).__init__()
        
        self.conv = Conv2d(nc, nf, filter_size, stride, padding, dilation, bias=bias_flag)
        self.bn = Norm2d(nf)
        
    def forward(self, x):
        x = activation_func(self.conv(x))
        x = self.bn(x)
        
        return x


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
            nn.activation_func,
            # state size. (ngf*2) x 14 x 14
            nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            nn.activation_func,
            # state size. (ngf*2) x 28 x 28
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            nn.activation_func,
            # state size. (ngf) x 56 x 56
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            nn.activation_func,
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