import torch
from torch.nn import Linear
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import math
import numbers

def calStructreLoss(velocity, migration, sobel_x, sobel_y, structre_measureLoss):
    grad_y_vel  = F.conv2d(velocity[None,None,...], sobel_y, padding = 'same') / 4
    grad_x_vel  = F.conv2d(velocity[None,None,...], sobel_x, padding = 'same') / 4
    structure_loss = structre_measureLoss(grad_y_vel, migration[None,None,...]) + structre_measureLoss(grad_x_vel, migration[None,None,...])
    return structure_loss

class NN(torch.nn.Module):
    def __init__(self, nl=10, activation=torch.nn.ELU(), input_num = 4, scale = 5, outputfunc = torch.nn.Sigmoid()):
            super(NN, self).__init__()
            self.act = activation

            # Input Structure
            self.fc0  = Linear(input_num,32)
            self.fc1  = Linear(32,512)

            # Resnet Block
            self.rn_fc1 = torch.nn.ModuleList([Linear(512, 512) for i in range(nl)])
            self.rn_fc2 = torch.nn.ModuleList([Linear(512, 512) for i in range(nl)])
            self.rn_fc3 = torch.nn.ModuleList([Linear(512, 512) for i in range(nl)])

            # Output structure
            self.fc8  = Linear(512,32)
            self.fc9  = Linear(32,1)
            self.outputfunc = outputfunc
            self.scale = scale

    def forward(self,x):
        x   = self.act(self.fc0(x))
        x   = self.act(self.fc1(x))
        for ii in range(len(self.rn_fc1)):
            x0 = x
            x  = self.act(self.rn_fc1[ii](x))
            x  = self.act(self.rn_fc3[ii](x)+self.rn_fc2[ii](x0))

        x   = self.act(self.fc8(x))
        tau = self.fc9(x)
        tau = self.outputfunc(tau) * self.scale
        return tau

class PINN(torch.nn.Module):
    def __init__(self, n1ForTime = 10, n1ForVel = 10, activation = torch.nn.ELU(), input_num_for_time = 4, input_num_for_vel = 2, scaleTime = 5, scaleVel = 5, outputfunc_time = torch.nn.Sigmoid(), outputfunc_vel = torch.nn.Sigmoid()):
        super(PINN, self).__init__()
        self.NNForTime = NN(nl = n1ForTime, activation = activation, input_num = input_num_for_time, scale = scaleTime, outputfunc = outputfunc_time)
        self.NNForVel = NN(nl = n1ForVel, activation = activation, input_num = input_num_for_vel, scale = scaleVel, outputfunc = outputfunc_vel)

    def forward(self, input_time_src, input_vel_corr):
        if input_time_src is not None:
            time_src = self.NNForTime(input_time_src)
        else:
            time_src = None
        vel_update = self.NNForVel(input_vel_corr)
        return time_src, vel_update
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def test_forward(self, input_time_Src):
        return self.NNForTime(input_time_Src)
    
    def velocity_forwad(self, input_vel_src):
        return self.NNForVel(input_vel_src)

class SCPINN(torch.nn.Module):
    def __init__(self, n1ForTime = 10, n1ForVel = 10, activation = torch.nn.ELU(), input_num_for_time = 4, input_num_for_vel = 2, scaleTime = 5, scaleVel = 5, outputfunc_time = torch.nn.Sigmoid(), outputfunc_vel = torch.nn.Tanh(), input_nc = 2, output_nc = 1, ngf = 64, norm = 'instance', tanhoutput = False):
        super(SCPINN, self).__init__()
        self.NNForTime = NN(nl = n1ForTime, activation = activation, input_num = input_num_for_time, scale = scaleTime, outputfunc = outputfunc_time)
        self.NNForVel = NN(nl = n1ForVel, activation = activation, input_num = input_num_for_vel, scale = scaleVel, outputfunc = outputfunc_vel)
        norm_layer = get_norm_layer(norm_type=norm)
        self.netG = InterpolateUnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=False, tanhoutput=tanhoutput)
    
    def forward(self, INPUTIMGTEST, velmean, vel_std, timesrcinput = None, velinput = None):
        dvgvelocity_all = self.DVGforward(INPUTIMGTEST, velmean, vel_std)
        if timesrcinput is not None:
            time_src = self.NNTimeforward(timesrcinput)
        else:
            time_src = None
        nnoutputvel = self.NNVelforward(velinput)
        return time_src, nnoutputvel, dvgvelocity_all
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def NNVelforward(self, input_vel_corr):
        return self.NNForVel(input_vel_corr)
    
    def NNTimeforward(self, input_time_src):
        return self.NNForTime(input_time_src)
    
    def DVGforward(self, input_img, vel_mean, vel_std):
        input_img, pad_img = pad(input_img)
        output_img = self.netG(input_img)
        output_img = unpad(output_img, *pad_img)
        outputvel = output_img * vel_std + vel_mean
        return outputvel
    
    def veltest_forward(self, input_vel_corr, input_img, vel_mean, vel_std):
        nnvel = self.NNVelforward(input_vel_corr)
        dvgvel = self.DVGforward(input_img, vel_mean, vel_std)
        return nnvel, dvgvel
    
    def velocity_forwad(self, input_vel_src, input_img, zcorrind, xcorrind, vel_mean, vel_std):
        vel_update = self.NNForVel(input_vel_src)
        output_img = self.netG(input_img)
        outpuutvel = output_img[:,:,zcorrind,xcorrind].view(-1,1)
        outpuutvel = outpuutvel * vel_std + vel_mean
        outpuutvel = outpuutvel + vel_update
        return outpuutvel

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance',track_running_stats = False):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=track_running_stats)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=track_running_stats)
        
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 1:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    else:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
    init_weights(net, init_type, init_gain=init_gain)
    return net
'''
def mrudsr_model(input_nc, output_nc, mid_channels, num_blocks, init_type = 'normal', init_gain = 0.02, gpu_ids = []):
    """
    create a m_rudsr model
    """
    net = MRUDSR(in_channels = input_nc, out_channels = output_nc, mid_channels = mid_channels, num_blocks = num_blocks)
    return init_net(net, init_type, init_gain, gpu_ids)
'''
def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], tanhoutput = False, vmin = None, vmax = None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout, tanhoutput=tanhoutput)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, tanhoutput=tanhoutput)
    elif netG == 'unet_128_s1':
        net = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout, tanhoutput=tanhoutput, stride_size = 1)
    elif netG == 'InterpolateUnet_256':
        net = InterpolateUnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, tanhoutput=tanhoutput)
    elif netG == 'VelocityGenerator':
        net = VelocityGeneratorSigmoid(input_nc, output_nc, 8, vmin, vmax, ngf, norm_layer=norm_layer, use_dropout=use_dropout, tanhoutput=tanhoutput)
    elif netG == 'SlopeInterpolateUnet2D':
        net = SlopeTomoModule(input_nc, output_nc, ngf = ngf, tanhoutput = tanhoutput)
    elif netG == 'SlopeInterpolateUnet3D':
        net = SlopeTomoModule(input_nc, output_nc, ngf = ngf, tanhoutput = tanhoutput, input_type = '3d')
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x): # x shape : N,C,H,W
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = (x.size(2)-1) * x.size(3)
        count_w = x.size(2) * (x.size(3)-1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]), 2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/x.size(0)

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, tanhoutput=False, stride_size = 2):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, stride_size = stride_size)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, stride_size = stride_size)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, stride_size = stride_size)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, stride_size = stride_size)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, stride_size = stride_size)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, tanhoutput=tanhoutput, stride_size = stride_size)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
# @torchsnooper.snoop()
class InterpolateUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, tanhoutput=False, stride_size = 2) -> None:
        super().__init__()
        '''
        Construct A Interpolate Unet Generator
        '''
        # construct unet structure
        unet_block = InterpolateUnetConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, stride_size = stride_size)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = InterpolateUnetConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, stride_size = stride_size)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = InterpolateUnetConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, stride_size = stride_size)
        unet_block = InterpolateUnetConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, stride_size = stride_size)
        unet_block = InterpolateUnetConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, stride_size = stride_size)
        self.model = InterpolateUnetConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, tanhoutput=tanhoutput, stride_size = stride_size)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class VelocityGeneratorSigmoid(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, vmin, vmax, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, tanhoutput=False, stride_size = 2) -> None:
        super().__init__()
        '''
        Construct A Interpolate Unet Generator
        '''
        # construct unet structure
        unet_block = InterpolateUnetConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, stride_size = stride_size)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = InterpolateUnetConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, stride_size = stride_size)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = InterpolateUnetConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, stride_size = stride_size)
        unet_block = InterpolateUnetConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, stride_size = stride_size)
        unet_block = InterpolateUnetConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, stride_size = stride_size)
        self.model = InterpolateUnetConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, tanhoutput=tanhoutput, stride_size = stride_size)  # add the outermost layer
        self.max_vel = vmax
        self.min_vel = vmin

    def forward(self, input):
        """Standard forward"""
        x = self.model(input)
        return torch.sigmoid(x)*(self.max_vel - self.min_vel) + self.min_vel

class InterpolateUnetConnectionBlock(nn.Module):
    '''
    Define Unet Connection Block With Interpolate Upsampling to escape checkboard artifacts
    X -------------------identity----------------------
        |-- downsampling -- |submodule| -- billinear interpolate -- conv2d --|
    '''
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False, tanhoutput = False, stride_size = 2) -> None:
        super().__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=stride_size, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc) 

        if outermost:
            upconv = nn.Conv2d(inner_nc * 2, outer_nc,
                                        kernel_size=3,padding = 'same') 
            upsample = nn.Upsample(scale_factor=2,align_corners=True,mode='bilinear')

            down = [downconv]
            if tanhoutput:
                up = [upsample, uprelu, upconv, nn.Tanh()]
            else:
                up = [upsample, uprelu, upconv]

            model = down + [submodule] + up
            
        elif innermost:
            upconv = nn.Conv2d(inner_nc, outer_nc,
                                        kernel_size=3, padding = 'same', bias=use_bias)
            down = [downrelu, downconv]
            upsample = nn.Upsample(scale_factor=2,align_corners=True,mode='bilinear')
            up = [upsample, uprelu, upconv, upnorm]
            model = down + up
            
        else:
            upconv = nn.Conv2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, padding = 'same', bias=use_bias)
            upsample = nn.Upsample(scale_factor=2,align_corners=True,mode='bilinear')

            down = [downrelu, downconv, downnorm]

            up = [upsample, uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     raise ValueError(f'Input to model is nan or inf at {self.__class__.__name__}\t{x.shape}')
        if self.outermost:
            return self.model(x)
        else:  
            return torch.cat([x, self.model(x)], 1)
        
class SlopeTomoModule(nn.Module):
    '''

    '''
    def __init__(self, input_nc = 50, output_nc = 1, stride = 2, norm_layer=nn.InstanceNorm2d, ngf = 64, tanhoutput = False, input_type='2d') -> None:
        super().__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.input_type = input_type
        if input_type == '2d':
            self.vel_input = nn.Conv2d( in_channels=input_nc, out_channels = input_nc // 2, kernel_size = 4, stride = stride, padding = 1, bias = use_bias)
            self.tau_input = nn.Conv2d( in_channels=input_nc, out_channels = input_nc // 2, kernel_size = 4, stride = stride, padding = 1, bias = use_bias)
            inpo_nc = input_nc // 2 * 2
        elif input_type == '3d':
            self.vel_input = nn.Conv3d( in_channels = 1, out_channels = 1, kernel_size = (8,4,4), stride = (4,2,2), padding = (1,0,0), bias = use_bias)
            self.tau_input = nn.Conv3d( in_channels = 1, out_channels = 1, kernel_size = (8,4,4), stride = (4,2,2), padding = (1,0,0), bias = use_bias)
            self.cmp_input = nn.Conv3d( in_channels = 1, out_channels = 1, kernel_size = (8,4,4), stride = (4,2,2), padding = (1,0,0), bias = use_bias)
            inpo_nc = 3
        
        self.interpolate_Unet = InterpolateUnetGenerator(inpo_nc, 8, 7, ngf = ngf, norm_layer = norm_layer, use_dropout = False, tanhoutput = tanhoutput)
        self.up_sample = nn.Upsample(scale_factor=2,align_corners=True,mode='bilinear')
        self.out_layer1 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(8,16,3,padding='same'),
            norm_layer(16),
            nn.ReLU(True),
            nn.Conv2d(16,4,3,padding='same')
        )
        self.out_layer2 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(12,output_nc,3,padding='same')
        )

    def forward(self, input_slowness, input_tau, input_cmp):
        if self.input_type == '2d':
            input_slowness = self.vel_input(input_slowness)
            input_tau = self.tau_input(input_tau)
            input_fea = torch.cat([input_slowness, input_tau], dim=1)
        elif self.input_type == '3d':
            input_slowness = input_slowness.unsqueeze(dim = 1)
            input_tau = input_tau.unsqueeze(dim = 1)
            input_cmp = input_cmp.unsqueeze(dim = 1)

            input_slowness = self.vel_input(input_slowness)
            input_tau = self.tau_input(input_tau)
            input_cmp = self.cmp_input(input_cmp)
            
            input_slowness = torch.mean(input_slowness.squeeze(dim=1), dim = 1, keepdim=True)
            input_tau = torch.mean(input_tau.squeeze(dim=1), dim = 1, keepdim=True)
            input_cmp = torch.mean(input_cmp.squeeze(dim=1), dim = 1, keepdim=True)

            input_fea = torch.cat([input_slowness, input_tau, input_cmp], dim = 1)

        input_fea = self.interpolate_Unet(input_fea)
        input_fea = self.up_sample(input_fea)
        input_fea1 = self.out_layer1(input_fea)
        input_fea = self.out_layer2( torch.cat([input_fea, input_fea1], dim=1) )
        return input_fea

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False, tanhoutput = False, stride_size = 2):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=stride_size, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=stride_size,
                                        padding=1) 
            down = [downconv]
            if tanhoutput:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=stride_size,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=stride_size,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  
            return torch.cat([x, self.model(x)], 1)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        # print(f'input tensor shape is {input.shape}')
        # print(f'output tensor shape is {self.model(input).shape}')
        return self.model(input)

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        Usage:
            smoothing = GaussianSmoothing(3, 5, 1)
            input = torch.rand(1, 3, 100, 100)
            input = F.pad(input, (2, 2, 2, 2), mode='reflect')
            output = smoothing(input)
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

def pad(x):
    def floor_ceil(n):
        return math.floor(n), math.ceil(n)
    b, c, h, w = x.shape
    w_mult = ((w - 1) | 255) + 1
    h_mult = ((h - 1) | 255) + 1
    w_pad = floor_ceil((w_mult - w) / 2)
    h_pad = floor_ceil((h_mult - h) / 2)
    x = F.pad(x, w_pad + h_pad, mode = 'replicate')
    return x, (h_pad, w_pad, h_mult, w_mult)

def unpad(x, h_pad, w_pad, h_mult, w_mult):
    return x[..., h_pad[0]:h_mult - h_pad[1], w_pad[0]:w_mult - w_pad[1]]

def _gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class StructureMeasureLoss(nn.Module):
    def __init__(self, window_size = 11, device = None):
        super(StructureMeasureLoss, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = _create_window(window_size, self.channel)
        if device is not None:
            self.window = self.window.to(device)
    
    def _stml(self, input1, input2, window, channel):
        mu1 = F.conv2d(input1, window, padding = 0, groups = channel)
        mu2 = F.conv2d(input2, window, padding = 0, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)

        mu1_mu2 = mu1*mu2
        sigma1_sq = F.conv2d(input1*input1, window, padding = 0, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(input2*input2, window, padding = 0, groups = channel) - mu2_sq
        sigma12 = F.conv2d(input1*input2, window, padding = 0, groups = channel) - mu1_mu2
        c3 = 1e-4

        s = (sigma12 + c3) / (torch.sqrt(torch.abs(sigma1_sq))*torch.sqrt(torch.abs(sigma2_sq)) + c3)

        return s.mean()

    def forward(self, input1, input2):
        return self._stml(input1, input2, self.window, self.channel)

class VelocityModel(torch.nn.Module):
    def __init__(self, initial, min_vel, max_vel):
        super().__init__()
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.model = torch.nn.Parameter(
            torch.logit((initial - min_vel) /
                        (max_vel - min_vel))
        )

    def forward(self):
        return (torch.sigmoid(self.model) *
                (self.max_vel - self.min_vel) +
                self.min_vel)
