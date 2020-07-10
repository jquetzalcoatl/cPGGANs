from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.data.DataLoader

def save_model(MODEL, module, flag, alpha, epoch, i, SAVE_DIR = "/home/jovyan/GANs/trained_models_r/"):
#     SAVE_DIR = "/home/jovyan/GANs/trained_models_r/"
    date = datetime.now().__str__()
    date = date[:16].replace(':', '-').replace(' ', '-')
    path = os.path.join(SAVE_DIR, f'{module}-{flag}-alph-{i}-epoch-{epoch}.pt')
    torch.save(MODEL.cpu().state_dict(), path) # saving model
    MODEL.cuda() # moving model to GPU for further training

def save_model2(model, module, flag, optimizer, loss, epoch, i, SAVE_DIR = "/home/jovyan/GANs/trained_models_r/"):
#     SAVE_DIR = "/home/jovyan/GANs/trained_models_r/"
    date = datetime.now().__str__()
    date = date[:16].replace(':', '-').replace(' ', '-')
    PATH = os.path.join(SAVE_DIR, f'{module}-{flag}-alph-{i}-epoch-{epoch}-complete.pt')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)
#     torch.save(MODEL.cpu().state_dict(), path) # saving model
#     MODEL.cuda() # moving model to GPU for further training

def load_model(model, optimizer, PATH):
#     model = TheModelClass(*args, **kwargs)
#     optimizer = TheOptimizerClass(*args, **kwargs)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()
    
# def correlation(a,b):
#     torch.ones(a)*torch.
    
def get_data_loader(x: torch.Tensor, y: torch.Tensor, batch_size=5, numThreads = 1) -> torch.utils.data.DataLoader:
    """Fetches a DataLoader, which is built into PyTorch, and provides a
    convenient (and efficient) method for sampling.

    :param x: (torch.Tensor) inputs
    :param y: (torch.Tensor) labels
    :param batch_size: (int)
    """
    dataset = torch.utils.data.TensorDataset(x, y)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=numThreads, shuffle=True, batch_size=batch_size)

    return data_loader
    
    
class EqualizedLR(nn.Module):
    def __init__(self, layer):
        super().__init__()

        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        layer.bias.data.fill_(0)

        self.wscale = layer.weight.data.detach().pow(2.).mean().sqrt()
        layer.weight.data /= self.wscale

        self.layer = layer

    def forward(self, x):
        return self.layer(x * self.wscale)
    

class PixelWiseNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_square_mean = x.pow(2).mean(dim=1, keepdim=True)
        denom = torch.rsqrt(x_square_mean + 1e-8)
        return x * denom
    
class mySeq(nn.Module):
    def __init__(self, in_c, out_c, flag=1):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        
        if flag == 0:
            self.seq = nn.Sequential(
                EqualizedLR(nn.ConvTranspose2d(self.in_c, self.out_c, 4)),
                PixelWiseNorm(),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                EqualizedLR(nn.Conv2d(self.out_c, self.out_c, 3, 1, 1)),
                PixelWiseNorm(),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        else:
            self.seq = nn.Sequential(
                EqualizedLR(nn.Conv2d(self.in_c, self.out_c, 3, 1, 1)),
                PixelWiseNorm(),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                EqualizedLR(nn.Conv2d(self.out_c, self.out_c, 3, 1, 1)),
                PixelWiseNorm(),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        
        
    def forward(self, x):
        return self.seq(x)
    
class mySeqD(nn.Module):
    def __init__(self, in_c, out_c, nb_digits = 0, flag=1):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.nb_digits = nb_digits
        
        if flag == 0:
            self.seq = nn.Sequential(
                MiniBatchSTD(),
                EqualizedLR(nn.Conv2d(in_c + 1 + nb_digits , out_c, 3, 1, 1)),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                EqualizedLR(nn.Conv2d(in_c, out_c, 4, 1, 0)),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                Flatten(),
                EqualizedLR(nn.Linear(out_c, 1)),
            )
        else:
            self.seq = nn.Sequential(
                EqualizedLR(nn.Conv2d(in_c + nb_digits, out_c, 3, 1, 1)),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                EqualizedLR(nn.Conv2d(out_c, out_c, 3, 1, 1)),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.AvgPool2d(2)
            )

        
    def forward(self, x):
        return self.seq(x)
        
        
class ToRGBLayer(nn.Module):
    def __init__(self, in_c, rgb_channel=3):
        super().__init__()

        self.conv = nn.Sequential(
            EqualizedLR(nn.Conv2d(in_c, rgb_channel, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)
    
class FromRGBLayer(nn.Module):
    def __init__(self, out_c, rgb_channel):
        super().__init__()
        self.conv = EqualizedLR(nn.Conv2d(rgb_channel, out_c, 1))

    def forward(self, x):
        return self.conv(x)
    
class MiniBatchSTD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        std = torch.std(x).expand(x.shape[0], 1, *x.shape[2:])
        return torch.cat([x, std], dim=1)
    
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        n = 64
        nc = 3
        self.ngpu = ngpu
        self.seq0 = mySeq(512,512,0)   # 4x4
        self.torgb = ToRGBLayer(512, 3)
        
        self.seq1 = mySeq(512,512)   # 8x8
        self.torgb1 = ToRGBLayer(512, 3)
        
        self.seq2 = mySeq(512,512)    #16x16
        self.torgb2 = ToRGBLayer(512, 3)
        
        self.seq3 = mySeq(512,512)     #32x32
        self.torgb3 = ToRGBLayer(512, 3)
        
        self.seq4 = mySeq(512,256)    #64x64
        self.torgb4 = ToRGBLayer(256, 3)
        
        
    def forward(self, input, flag = [64, "stable"], alpha=0):
        if flag == [4, "stable"]:
            x = self.seq0(input)
            x = self.torgb(x)
        elif flag == [8, "transition"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x1 = self.torgb(x)
            x2 = self.seq1(x)
            x2 = self.torgb1(x2)
            x = x2 * alpha + (1.0 - alpha) * x1
        elif flag == [8, "stable"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq1(x)
            x = self.torgb1(x)
        elif flag == [16, "transition"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq1(x)
            x = F.interpolate(x, scale_factor=2)
            x1 = self.torgb1(x)
            x2 = self.seq2(x)
            x2 = self.torgb2(x2)
            x = x2 * alpha + (1.0 - alpha) * x1
        elif flag == [16, "stable"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq1(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq2(x)
            x = self.torgb2(x)
        elif flag == [32, "transition"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq1(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq2(x)
            x = F.interpolate(x, scale_factor=2)
            x1 = self.torgb2(x)
            x2 = self.seq3(x)
            x2 = self.torgb3(x2)
            x = x2 * alpha + (1.0 - alpha) * x1
        elif flag == [32, "stable"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq1(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq2(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq3(x)
            x = self.torgb3(x)
        elif flag == [64, "transition"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq1(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq2(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq3(x)
            x = F.interpolate(x, scale_factor=2)
            x1 = self.torgb3(x)
            x2 = self.seq4(x)
            x2 = self.torgb4(x2)
            x = x2 * alpha + (1.0 - alpha) * x1
        elif flag == [64, "stable"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq1(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq2(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq3(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq4(x)
            x = self.torgb4(x)

        return x
    
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()      
        n = 64
        nc = 3
        self.ngpu = ngpu
        self.seq0 = mySeqD(512,512,0)
        self.fromrgb = FromRGBLayer(512, 3)
        
        self.seq1 = mySeq(512,512)
        self.fromrgb1 = FromRGBLayer(512, 3)
        
        self.seq2 = mySeq(512,512)
        self.fromrgb2 = FromRGBLayer(512, 3)
        
        self.seq3 = mySeq(512,512)
        self.fromrgb3 = FromRGBLayer(512, 3)
        
        self.seq4 = mySeq(256,512)
        self.fromrgb4 = FromRGBLayer(256, 3)
        
    def forward(self, input, flag = [64, "stable"], alpha=0):
        if flag == [4, "stable"]:
            x = self.fromrgb(input)
            x = self.seq0(x)
        elif flag == [8, "transition"]:   #F.avg_pool2d(x, kernel_size=2)
            x1 = F.avg_pool2d(input, kernel_size=2)
            x1 = self.fromrgb(x1)
            x2 = self.fromrgb(input)
            x2 = self.seq1(x2)
            x2 = F.avg_pool2d(x2, kernel_size=2)
            x = x2 * alpha + (1.0 - alpha) * x1
            x = self.seq0(x)
        elif flag == [8, "stable"]:
            x = self.fromrgb(input)
            x = self.seq1(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq0(x)
        elif flag == [16, "transition"]:   #F.avg_pool2d(x, kernel_size=2)
            x1 = F.avg_pool2d(input, kernel_size=2)
            x1 = self.fromrgb(x1)
            x2 = self.fromrgb(input)
            x2 = self.seq2(x2)
            x2 = F.avg_pool2d(x2, kernel_size=2)
            x = x2 * alpha + (1.0 - alpha) * x1
            x = self.seq1(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq0(x)
        elif flag == [16, "stable"]:
            x = self.fromrgb(input)
            x = self.seq2(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq1(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq0(x)
        elif flag == [32, "transition"]:   
            x1 = F.avg_pool2d(input, kernel_size=2)
            x1 = self.fromrgb(x1)
            x2 = self.fromrgb(input)
            x2 = self.seq3(x2)
            x2 = F.avg_pool2d(x2, kernel_size=2)
            x = x2 * alpha + (1.0 - alpha) * x1
            x = self.seq2(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq1(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq0(x)
        elif flag == [32, "stable"]:
            x = self.fromrgb(input)
            x = self.seq3(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq2(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq1(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq0(x)
        elif flag == [64, "transition"]:   
            x1 = F.avg_pool2d(input, kernel_size=2)
            x1 = self.fromrgb3(x1)
            x2 = self.fromrgb4(input)
            x2 = self.seq4(x2)
            x2 = F.avg_pool2d(x2, kernel_size=2)
            x = x2 * alpha + (1.0 - alpha) * x1
            x = self.seq3(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq2(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq1(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq0(x)
        elif flag == [64, "stable"]:
            x = self.fromrgb4(input)
            x = self.seq4(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq3(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq2(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq1(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = self.seq0(x)

        return x
    
class GeneratorC(nn.Module):
    def __init__(self, ngpu, nb_digits=5):
        super(GeneratorC, self).__init__()
        n = 64
        nc = 3
        self.ngpu = ngpu
        self.seq0 = mySeq(512 + nb_digits,512,0)   # 4x4
        self.torgb = ToRGBLayer(512, 3)
        
        self.seq1 = mySeq(512,512)   # 8x8
        self.torgb1 = ToRGBLayer(512, 3)
        
        self.seq2 = mySeq(512,512)    #16x16
        self.torgb2 = ToRGBLayer(512, 3)
        
        self.seq3 = mySeq(512,512)     #32x32
        self.torgb3 = ToRGBLayer(512, 3)
        
        self.seq4 = mySeq(512,256)    #64x64
        self.torgb4 = ToRGBLayer(256, 3)
        
        
    def forward(self, input, flag = [64, "stable"], alpha=0):
        if flag == [4, "stable"]:
            x = self.seq0(input)
            x = self.torgb(x)
        elif flag == [8, "transition"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x1 = self.torgb(x)
            x2 = self.seq1(x)
            x2 = self.torgb1(x2)
            x = x2 * alpha + (1.0 - alpha) * x1
        elif flag == [8, "stable"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq1(x)
            x = self.torgb1(x)
        elif flag == [16, "transition"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq1(x)
            x = F.interpolate(x, scale_factor=2)
            x1 = self.torgb1(x)
            x2 = self.seq2(x)
            x2 = self.torgb2(x2)
            x = x2 * alpha + (1.0 - alpha) * x1
        elif flag == [16, "stable"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq1(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq2(x)
            x = self.torgb2(x)
        elif flag == [32, "transition"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq1(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq2(x)
            x = F.interpolate(x, scale_factor=2)
            x1 = self.torgb2(x)
            x2 = self.seq3(x)
            x2 = self.torgb3(x2)
            x = x2 * alpha + (1.0 - alpha) * x1
        elif flag == [32, "stable"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq1(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq2(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq3(x)
            x = self.torgb3(x)
        elif flag == [64, "transition"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq1(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq2(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq3(x)
            x = F.interpolate(x, scale_factor=2)
            x1 = self.torgb3(x)
            x2 = self.seq4(x)
            x2 = self.torgb4(x2)
            x = x2 * alpha + (1.0 - alpha) * x1
        elif flag == [64, "stable"]:
            x = self.seq0(input)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq1(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq2(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq3(x)
            x = F.interpolate(x, scale_factor=2)
            x = self.seq4(x)
            x = self.torgb4(x)

        return x
    
class DiscriminatorC(nn.Module):
    def __init__(self, ngpu, nb_digits = 5):
        super(DiscriminatorC, self).__init__()      
        n = 64
        nc = 3
        self.nb_digits = nb_digits
        self.ngpu = ngpu
        self.seq0 = mySeqD(512, 512, nb_digits, 0)
        self.fromrgb = FromRGBLayer(512, 3)
        
        self.seq1 = mySeq(512 + nb_digits,512)
#         self.fromrgb1 = FromRGBLayer(512, 3)
        
        self.seq2 = mySeq(512 + nb_digits,512)
#         self.fromrgb2 = FromRGBLayer(512, 3)
        
        self.seq3 = mySeq(512 + nb_digits,512)
#         self.fromrgb3 = FromRGBLayer(512, 3)
        
        self.seq4 = mySeq(256 + nb_digits,512)
        self.fromrgb4 = FromRGBLayer(256, 3 )
        
    def turn_onehot_to_block(self, y, size):     
        #Receives onehot, turns them to blocks of one-hot. This block
        # should be concat w/ x after the first convolution in the D.
        y_onehot_rep = y.expand(size[2],size[3],y.size()[0],self.nb_digits).permute(2,3,0,1)
        return y_onehot_rep
              
    def forward(self, input, y, flag = [64, "stable"], alpha=0):
#         input = torch.cat((input,self.turn_onehot_to_block(y,input.size())), dim=1)
        if flag == [4, "stable"]:
            x = self.fromrgb(input)
            x = torch.cat((x,self.turn_onehot_to_block(y, x.size())), dim=1)
            x = self.seq0(x)
        elif flag == [8, "transition"]:   #F.avg_pool2d(x, kernel_size=2)
            x1 = F.avg_pool2d(input, kernel_size=2)
            x1 = self.fromrgb(x1)
#             x1 = torch.cat((x1,self.turn_onehot_to_block(y,x1.size())), dim=1)
            x2 = self.fromrgb(input)
            x2 = torch.cat((x2,self.turn_onehot_to_block(y,x2.size())), dim=1)
            x2 = self.seq1(x2)
            x2 = F.avg_pool2d(x2, kernel_size=2)
            x = x2 * alpha + (1.0 - alpha) * x1
            x = torch.cat((x,self.turn_onehot_to_block(y, x.size())), dim=1)
            x = self.seq0(x)
        elif flag == [8, "stable"]:
            x = self.fromrgb(input)
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq1(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y, x.size())), dim=1)
            x = self.seq0(x)
        elif flag == [16, "transition"]:   #F.avg_pool2d(x, kernel_size=2)
            x1 = F.avg_pool2d(input, kernel_size=2)
            x1 = self.fromrgb(x1)
#             x1 = torch.cat((x1,self.turn_onehot_to_block(y,x1.size())), dim=1)
            x2 = self.fromrgb(input)
            x2 = torch.cat((x2,self.turn_onehot_to_block(y,x2.size())), dim=1)
            x2 = self.seq2(x2)
            x2 = F.avg_pool2d(x2, kernel_size=2)
            x = x2 * alpha + (1.0 - alpha) * x1
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq1(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y, x.size())), dim=1)
            x = self.seq0(x)
        elif flag == [16, "stable"]:
            x = self.fromrgb(input)
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq2(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq1(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y, x.size())), dim=1)
            x = self.seq0(x)
        elif flag == [32, "transition"]:   
            x1 = F.avg_pool2d(input, kernel_size=2)
            x1 = self.fromrgb(x1)
#             x1 = torch.cat((x1,self.turn_onehot_to_block(y,x1.size())), dim=1)
            x2 = self.fromrgb(input)
            x2 = torch.cat((x2,self.turn_onehot_to_block(y,x2.size())), dim=1)
            x2 = self.seq3(x2)
            x2 = F.avg_pool2d(x2, kernel_size=2)
            x = x2 * alpha + (1.0 - alpha) * x1
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq2(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq1(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y, x.size())), dim=1)
            x = self.seq0(x)
        elif flag == [32, "stable"]:
            x = self.fromrgb(input)
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq3(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq2(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq1(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y, x.size())), dim=1)
            x = self.seq0(x)
        elif flag == [64, "transition"]:   
            x1 = F.avg_pool2d(input, kernel_size=2)
            x1 = self.fromrgb(x1)
#             x1 = torch.cat((x1,self.turn_onehot_to_block(y,x1.size())), dim=1)
            x2 = self.fromrgb4(input)
            x2 = torch.cat((x2,self.turn_onehot_to_block(y,x2.size())), dim=1)
            x2 = self.seq4(x2)
            x2 = F.avg_pool2d(x2, kernel_size=2)
            x = x2 * alpha + (1.0 - alpha) * x1
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq3(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq2(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq1(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y, x.size())), dim=1)
            x = self.seq0(x)
        elif flag == [64, "stable"]:
            x = self.fromrgb4(input)
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq4(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq3(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq2(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y,x.size())), dim=1)
            x = self.seq1(x)
            x = F.avg_pool2d(x, kernel_size=2)
            x = torch.cat((x,self.turn_onehot_to_block(y, x.size())), dim=1)
            x = self.seq0(x)

        return x