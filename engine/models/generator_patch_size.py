import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)


class Translator(nn.Module):
    def __init__(self, ngf=64, img_size=32, nc=3):
        super(Translator, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(nc, ngf, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True), 
            # 16 x 16
            nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 x 8

            nn.Conv2d(ngf*2, ngf*4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf*4, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            #16x16

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 x 32

            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),   
        )

    def forward(self, x):
        img = self.conv_blocks(x)
        return img


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 4 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            #nn.Conv2d(ngf*8, ngf*4, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*4, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),   
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class CondGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3, num_classes=100):
        super(CondGenerator, self).__init__()
        self.num_classes = num_classes
        self.emb = nn.Embedding(num_classes, nz)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(2*nz, ngf * 4 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 4),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf*4, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),   
        )

    def interpolate(self, y1, y2, alpha):
        y1 = self.emb(y1)
        y2 = self.emb(y2)
        y = alpha * y1 + (1-alpha) * y2
        return y

    def forward(self, z, y, embedded_y=False):
        if not embedded_y:
            y = self.emb(y)
        out = self.l1(torch.cat([z, y], dim=1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class ResnetBlock(nn.Module):
    def __init__(self,
                 fin,
                 fout,
                 fhidden=None,
                 is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden
        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden,
                      self.fout,
                      3,
                      stride=1,
                      padding=1,
                      bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin,
                          self.fout,
                          1,
                          stride=1,
                          padding=0,
                          bias=False)
        self.bn0 = nn.BatchNorm2d(self.fin)
        self.bn1 = nn.BatchNorm2d(self.fhidden)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(F.leaky_relu(self.bn0(x)))
        dx = self.conv_1(F.leaky_relu(self.bn1(dx)))
        out = x_s + 0.1 * dx
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class DCGAN_Generator(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """
    def __init__(self, nz=100, ngf=64, nc=3, img_size=64, slope=0.2):
        super(DCGAN_Generator, self).__init__()
        self.nz = nz
        if isinstance(img_size, (list, tuple)):
            self.init_size = ( img_size[0]//16, img_size[1]//16 )
        else:    
            self.init_size = ( img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz, ngf*8*self.init_size[0]*self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf*8),
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x

            nn.Conv2d(ngf, nc, 3, 1,1),
            nn.Sigmoid(),
            #nn.Sigmoid()
        )

    def forward(self, z):
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output

class DCGAN_CondGenerator(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """
    def __init__(self, num_classes,  nz=100, n_emb=50, ngf=64, nc=3, img_size=64, slope=0.2):
        super(DCGAN_CondGenerator, self).__init__()
        self.nz = nz
        self.emb = nn.Embedding(num_classes, n_emb)
        if isinstance(img_size, (list, tuple)):
            self.init_size = ( img_size[0]//16, img_size[1]//16 )
        else:    
            self.init_size = ( img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz+n_emb, ngf*8*self.init_size[0]*self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf*8),
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x

            nn.Conv2d(ngf, nc, 3, 1,1),
            #nn.Tanh(),
            nn.Sigmoid()
        )

    def forward(self, z, y):
        y = self.emb(y)
        z = torch.cat([z, y], dim=1)
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output

class Discriminator(nn.Module):
    def __init__(self, nc=3, img_size=32, ndf=64):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *discriminator_block(nc, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

class DCGAN_Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(DCGAN_Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),
            #nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class PatchDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=128):
        super(PatchDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, 1, 1, 1, 0, bias=False),
        )
    
    def forward(self, input):
        return self.main(input)


class InceptionDiscriminator(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(InceptionDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(hidden_channel, 1, kernel_size=1, bias=False),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

class PatchDiscriminator_psize1(nn.Module):
    def __init__(self, nc=3, ndf=128):
        super(PatchDiscriminator_psize1, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, 1, 1, 1, 0, bias=False),
        )
    
    def forward(self, input):
        return self.main(input)

class PatchDiscriminator_psize2(nn.Module):
    def __init__(self, nc=3, ndf=128):
        super(PatchDiscriminator_psize2, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 2, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, 1, 1, 1, 0, bias=False),
        )
    
    def forward(self, input):
        return self.main(input)

class PatchDiscriminator_psize4(nn.Module):
    def __init__(self, nc=3, ndf=128):
        super(PatchDiscriminator_psize4, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf *2, 1, 1, 1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, 1, 1, 1, 0, bias=False),
        )
    
    def forward(self, input):
        return self.main(input)

class PatchDiscriminator_psize10(nn.Module):
    def __init__(self, nc=3, ndf=128):
        super(PatchDiscriminator_psize10, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 1, 1, 0, bias=False),
        )
    def forward(self, input):
        return self.main(input)

class PatchDiscriminator_psize18(nn.Module):
    def __init__(self, nc=3, ndf=128):
        super(PatchDiscriminator_psize18, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 3, 1, 1, bias=False),
        )
    
    def forward(self, input):
        return self.main(input)

class PatchDiscriminator_psize22(nn.Module):
    def __init__(self, nc=3, ndf=128):
        super(PatchDiscriminator_psize22, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf* 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 1, 1, 0, bias=False),
        )
    
    def forward(self, input):
        return self.main(input)