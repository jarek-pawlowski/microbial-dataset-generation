#---------------------------------------------------------------
#
# adapted from github.com/limingcv/Photorealistic-Style-Transfer
#
#---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import numpy as np
import time

img_mean = (0.485, 0.456, 0.406) # ImageNet
img_std = (0.229, 0.224, 0.225)

#img_mean = (0.406, 0.456, 0.485)
#img_std = (0.225, 0.224, 0.229)


IN_MOMENTUM = 0.15
gpu_id = "cuda:0"

class ReflectionConv(nn.Module):
    '''
        Reflection padding convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ReflectionConv, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        return out

class ConvLayer(nn.Module):
    '''
        zero-padding convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        conv_padding = int(np.floor(kernel_size / 2))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=conv_padding)
    def forward(self, x):
        return self.conv(x)
  
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU(inplace=True) # 1

        self.identity_block = nn.Sequential(
            ConvLayer(in_channels, out_channels // 4, kernel_size=1, stride=1),
            nn.InstanceNorm2d(out_channels // 4, momentum=IN_MOMENTUM),
            nn.ReLU(),
            ConvLayer(out_channels // 4, out_channels // 4, kernel_size, stride=stride),
            nn.InstanceNorm2d(out_channels // 4, momentum=IN_MOMENTUM),
            nn.ReLU(),
            ConvLayer(out_channels // 4, out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm2d(out_channels, momentum=IN_MOMENTUM),
            nn.ReLU(),
        )
        self.shortcut = nn.Sequential(
            ConvLayer(in_channels, out_channels, 1, stride),
            nn.InstanceNorm2d(out_channels)
        )
    
    def forward(self, x):
        out = self.identity_block(x)
        if self.in_channels == self.out_channels:
            residual = x
        else: 
            residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out

class Upsample(nn.Module):
    '''
        Since the number of channels of the feature map changes after upsampling in HRNet.
        we have to write a new Upsample class.
    '''
    def __init__(self, in_channels, out_channels, scale_factor, mode):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.instance = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.upsample(out)
        out = self.instance(out)
        out = self.relu(out)
        
        return out

class HRNet(nn.Module):
    def __init__(self):
        super(HRNet, self).__init__()

        self.pass1_1 = BasicBlock(3, 16, kernel_size=3, stride=1)
        self.pass1_2 = BasicBlock(16, 32, kernel_size=3, stride=1)
        self.pass1_3 = BasicBlock(32, 32, kernel_size=3, stride=1)
        self.pass1_4 = BasicBlock(64, 64, kernel_size=3, stride=1)
        self.pass1_5 = BasicBlock(192, 64, kernel_size=3, stride=1)
        self.pass1_6 = BasicBlock(64, 32, kernel_size=3, stride=1)
        self.pass1_7 = BasicBlock(32, 16, kernel_size=3, stride=1)
        self.pass1_8 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.pass2_1 = BasicBlock(32, 32, kernel_size=3, stride=1)
        self.pass2_2 = BasicBlock(64, 64, kernel_size=3, stride=1)
        
        self.downsample1_1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.downsample1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.downsample1_3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.downsample1_4 = nn.Conv2d(32, 32, kernel_size=3, stride=4, padding=1)
        self.downsample1_5 = nn.Conv2d(64, 64, kernel_size=3, stride=4, padding=1)
        self.downsample2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.downsample2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.upsample1_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample1_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2_1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample2_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        map1 = self.pass1_1(x)
        map2 = self.pass1_2(map1)
        map3 = self.downsample1_1(map1)
        map4 = torch.cat((self.pass1_3(map2), self.upsample1_1(map3)), 1)
        map5 = torch.cat((self.downsample1_2(map2), self.pass2_1(map3)), 1)
        map6 = torch.cat((self.downsample1_4(map2), self.downsample2_1(map3)), 1)
        map7 = torch.cat((self.pass1_4(map4), self.upsample1_2(map5), self.upsample2_1(map6)), 1)
        out = self.pass1_5(map7)
        out = self.pass1_6(out)
        out = self.pass1_7(out)
        out = self.pass1_8(out)

        return out

def load_image(image):
    ''' 
        Change image into tensor and normalize it
    ''' 
    #image = Image.open(image)
    image = Image.fromarray(image)
    transform = transforms.Compose([
                        # convert the (H x W x C) PIL image in the range(0, 255) into (C x H x W) tensor in the range(0.0, 1.0) 
                        transforms.ToTensor(),
                        transforms.Normalize(img_mean, img_std),   # this is from ImageNet dataset
                        ])   

    # change image's size to (b, 3, h, w)
    image = transform(image)[:3, :, :].unsqueeze(0)

    return image

def im_convert(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)    # change size to (channel, height, width)

    '''
        tensor (batch, channel, height, width)
        numpy.array (height, width, channel)
        to transform tensor to numpy, tensor.transpose(1,2,0) 
    '''
    image = image.transpose(1,2,0)
    image = image * np.array(img_std) + np.array(img_mean)   # change into unnormalized image
    #
    #for ch in range(3):
    #    low = np.amin(image[:,:,ch])
    #    if low < 0.: image[:,:,ch] -= low
    #for ch in range(3):
    #    high = np.amax(image[:,:,ch])
    #    if high > 1.: 
    #        image[:,:,ch] /= high
    #
    image = image.clip(0, 1)    # in the previous steps, we change PIL image(0, 255) into tensor(0.0, 1.0), so convert it
    return np.array(image)


def get_features(image, model, layers=None):
    '''
        return a dictionary consists of each layer's name and it's feature maps
    '''
    if layers is None:
        layers = {'0': 'conv1_1',   # default style layer
                  '5': 'conv2_1',   # default style layer
                  '10': 'conv3_1',  # default style layer
                  '19': 'conv4_1',  # default style layer
                  '21': 'conv4_2',  # default content layer
                  '28': 'conv5_1'}  # default style layer
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)    #  layer(x) is the feature map through the layer when the input is x
        if name in layers:
            features[layers[name]] = x
    
    return features


def get_grim_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, h * w)
    gram_matrix = torch.mm(tensor, tensor.t())
    return gram_matrix


def style_transfer(content_image, style_image, previous_model):

    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    print('------------------------------------------------------------------')
    print('style transfer using ' + str(device) + ' started...')
    print('------------------------------------------------------------------')

    # get the VGG19's structure except the full-connect layers
    VGG = models.vgg19(pretrained=True).features
    VGG.to(device)
    for parameter in VGG.parameters():
        parameter.requires_grad_(False)

    style_net = HRNet()
    if previous_model is not None:
        style_net.load_state_dict(previous_model)
    style_net.to(device)

    content_image = load_image(content_image)
    content_image = content_image.to(device)
    style_image = load_image(style_image)
    style_image = style_image.to(device)

    content_features = get_features(content_image, VGG)
    style_features   = get_features(style_image, VGG)

    style_gram_matrixs = {layer: get_grim_matrix(style_features[layer]) for layer in style_features}
    target = content_image.clone().requires_grad_(True).to(device)

    # try to give for con_layers more weight so that can get more detail in output image
    style_weights = {'conv1_1': 0.1,
                     'conv2_1': 0.2,
                     'conv3_1': 0.4,
                     'conv4_1': 0.8, # 0.8
                     'conv5_1': 1.6} # 1.6

    content_weight = 150
    style_weight = 0.7  # 0.5-5

    optimizer = optim.Adam(style_net.parameters(), lr=5e-3) # 5e-3
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    steps = 500  # 500

    content_loss_epoch = []
    style_loss_epoch = []
    total_loss_epoch = []
    output_image = content_image

    time_start=time.time()
    for epoch in range(0, steps+1):
        
        scheduler.step()
        target = style_net(content_image).to(device)
        target.requires_grad_(True)
        target_features = get_features(target, VGG)  # extract output image's all feature maps
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)    
        style_loss = 0
        # compute each layer's style loss and add them
        for layer in style_weights:
           
            target_feature = target_features[layer]  # output image's feature map after layer
            target_gram_matrix = get_grim_matrix(target_feature)
            style_gram_matrix = style_gram_matrixs[layer]

            layer_style_loss = style_weights[layer] * torch.mean((target_gram_matrix - style_gram_matrix) ** 2)
            b, c, h, w = target_feature.shape
            style_loss += layer_style_loss / (c * h * w)
        
        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss_epoch.append(total_loss)
        style_loss_epoch.append(style_weight * style_loss)
        content_loss_epoch.append(content_weight * content_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        output_image = target
        if (epoch+1) % 50 == 0: print(str(epoch+1) + " epochs elapsed...")
    style_net.eval().cpu()
    time_end=time.time()
    print('style transfer time cost (s)', int(time_end - time_start))

    return im_convert(target), style_net.state_dict()
