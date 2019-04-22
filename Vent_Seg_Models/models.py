from fastai.vision import *
import math

__all__ = ['MeshNet', 'VolumetricUnet', 'conv_relu_bn_drop', 'res3dmodel', 'get_total_params',
           'VolumetricResidualUnet', 'model_dict', 'experiment_model_dict', 'one_by_one_conv',
           'model_split_dict']


####################
   ## GET MODELS ##
####################

# 1 - default unet
def unet_default(**kwargs):
    'https://arxiv.org/pdf/1606.06650.pdf'
    return VolumetricUnet(in_c=1, out_c=4, n_layers=3, c=1, block_type=conv_relu_bn_drop, **kwargs)

# 2 - unet wider
def unet_wide(**kwargs):
    return VolumetricUnet(in_c=1, out_c=8, n_layers=3, c=1, block_type=conv_relu_bn_drop, **kwargs)

# 3 - unet deeper
def unet_deep(**kwargs):
    return VolumetricUnet(in_c=1, out_c=4, n_layers=5, c=1, block_type=conv_relu_bn_drop, **kwargs)

# 4 - unet wide deep
def unet_wide_deep(**kwargs):
    return VolumetricUnet(in_c=1, out_c=8, n_layers=5, c=1, block_type=conv_relu_bn_drop, **kwargs)

# 5 - default meshnet - dilated FCN
def meshnet():
    return MeshNet(in_c=1, out_c=8, num_classes=1, drop=0, dilations=[1,1,1,2,4,8,1,1], kernel_size=3)

# 6 - default modified 3d unet - lower lr=1e-1
def modified_unet():
    return Modified3DUNet(in_channels=1, n_classes=1, base_n_filter = 8)

# 6b - wider modified 3d unet
def modified_unet_wide():
    return Modified3DUNet(in_channels=1, n_classes=1, base_n_filter = 8)

# 7 - 3d residual model
def res3d():
    return res3dmodel()

# residual fused unet
def residual_unet():
    'https://arxiv.org/pdf/1802.10508.pdf'
    return VolumetricResidualUnet(in_c=8, p=0.2, norm_type='instance', actn='prelu')

# residual fused unet wide
def residual_unet_wide():
    'https://arxiv.org/pdf/1802.10508.pdf'
    return VolumetricResidualUnet(in_c=12, p=0.2, norm_type='instance', actn='prelu')

# residual fused unet
def residual_unet_v2():
    'https://arxiv.org/pdf/1802.10508.pdf'
    return VolumetricResidualUnet(in_c=8, p=0.2, norm_type='batch', actn='relu')

# residual fused unet wide
def residual_unet_wide_v2():
    'https://arxiv.org/pdf/1802.10508.pdf'
    return VolumetricResidualUnet(in_c=12, p=0.2, norm_type='batch', actn='relu')

model_dict = {
    'unet_default': unet_default,
    'unet_wide': unet_wide,
    'unet_deep': unet_deep,
    'unet_wide_deep': unet_wide_deep,
    'meshnet': meshnet,
    'modified_unet': modified_unet,
    'modified_unet_wide': modified_unet_wide,
    'res3d': res3d,
    'residual_unet': residual_unet,
    'residual_unet_wide': residual_unet_wide
}

experiment_model_dict = {
    'baseline1': partial(unet_default, p=0., norm_type='batch', actn='relu'), # bce
    'baseline2': partial(unet_default, p=0., norm_type='batch', actn='relu'), # dice
    'baseline3': partial(unet_default, p=0., norm_type='group', actn='relu'),
    'baseline4': partial(unet_default, p=0., norm_type='group', actn='prelu'),
    'baseline5': partial(unet_default, p=0.3, norm_type='group', actn='prelu'),
    'baseline6': meshnet,
    'baseline7': partial(unet_wide, p=0., norm_type='group', actn='prelu'),
    'baseline8': partial(unet_deep, p=0., norm_type='group', actn='prelu'),
    'baseline9': partial(unet_wide_deep, p=0., norm_type='group', actn='prelu'),
    'baseline10': residual_unet,
    'baseline11': residual_unet_wide,
    'baseline10_v2': residual_unet_v2,
    'baseline11_v2': residual_unet_wide_v2,
}

####################
  ## SPLIT FUNCS ##
####################

def _baseline1_split(m:nn.Module): return (nn.ModuleList([m.downblocks,m.middle]), 
                                           m.upblocks,
                                           m.conv_final)

def _baseline6_split(m:nn.Module): return (m.layers[:4], m.layers[4:6], m.layers[7:])

def _baseline8_split(m:nn.Module): return (nn.ModuleList([m.downblocks, m.middle]),
                                            m.upblocks[:3],
                                            m.upblocks[3:],
                                            m.conv_final)

def _baseline10_split(m:nn.Module): return (nn.ModuleList([m.down1, m.down2, m.down3, m.down4]),
                                           nn.ModuleList([m.middle, m.upblock1]),
                                           nn.ModuleList([m.upblock2, m.upblock3, m.seg2, m.seg3, m.seg_final])
                                          )

model_split_dict = {
    'baseline1': _baseline1_split,
    'baseline2': _baseline1_split,
    'baseline3': _baseline1_split,
    'baseline4': _baseline1_split,
    'baseline5': _baseline1_split,
    'baseline6': _baseline6_split,
    'baseline7': _baseline1_split,
    'baseline8': _baseline8_split,
    'baseline9': _baseline8_split,
    'baseline10': _baseline10_split,
    'baseline11': _baseline10_split,
}


####################
    ## LAYERS ##
####################

class RunningBatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom, self.eps = mom, eps
        self.mults = nn.Parameter(torch.ones (nf,1,1))
        self.adds  = nn.Parameter(torch.zeros(nf,1,1))
        self.register_buffer('sums', torch.zeros(1,nf,1,1))
        self.register_buffer('sqrs', torch.zeros(1,nf,1,1))
        self.register_buffer('count', tensor(0.))
        self.register_buffer('factor', tensor(0.))
        self.register_buffer('offset', tensor(0.))
        self.batch = 0
        
    def update_stats(self, x):
        bs,nc,*_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0,2,3)
        s    = x    .sum(dims, keepdim=True)
        ss   = (x*x).sum(dims, keepdim=True)
        c    = s.new_tensor(x.numel()/nc)
        mom1 = s.new_tensor(1 - (1-self.mom)/math.sqrt(bs-1))
        self.sums .lerp_(s , mom1)
        self.sqrs .lerp_(ss, mom1)
        self.count.lerp_(c , mom1)
        self.batch += bs
        means = self.sums/self.count
        varns = (self.sqrs/self.count).sub_(means*means)
        if bool(self.batch < 20): varns.clamp_min_(0.01)
        self.factor = self.mults / (varns+self.eps).sqrt()
        self.offset = self.adds - means*self.factor
        
    def forward(self, x):
        if self.training: self.update_stats(x)
        return x*self.factor + self.offset
    
def get_total_params(model):
    params = model.parameters()
    tot_params = 0
    for p in params:
        prod = np.product(p.shape)
        tot_params += prod
        print(p.shape, prod)
    print('total:', tot_params)
    return tot_params

def maxpool3D(): return nn.MaxPool3d(2, stride=2)

def one_by_one_conv(in_channel, out_channel): return nn.Conv3d(in_channel, out_channel, 1)

def conv_relu_bn_drop(in_channel, out_channel, dilation=1, p=0.5, norm_type='batch', actn='relu', init_func=None, stride=1, kernel=3):
    'conv (pad=dilation) - same padding -> relu -> norm -> dropout'
    if norm_type == 'batch': norm = nn.BatchNorm3d(out_channel)
    if norm_type == 'instance': norm = nn.InstanceNorm3d(out_channel)
    if norm_type == 'group': norm = nn.GroupNorm(2, out_channel)
    if norm_type == 'running': norm = RunningBatchNorm(out_channel)

    if actn == 'relu': actn_fn = nn.ReLU(inplace=True)
    if actn == 'prelu': actn_fn = nn.PReLU()

    conv = nn.Conv3d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=dilation, dilation=dilation, bias=True)
    if init_func: init_default(conv, init_func)
    drop = nn.Dropout3d(p)
    return nn.Sequential(conv, actn_fn, norm, drop)

####################
    ## MESHNET ##
####################

class MeshNet(nn.Module):
#     https://arxiv.org/pdf/1612.00940.pdf
    def __init__(self, in_c=1, out_c=24, num_classes=1, drop=0, 
                 dilations=[1,1,1,2,4,8,1,1], kernel_size=3):
        super(MeshNet, self).__init__()
        self.layers = []
        n = len(dilations[1:-1])
        self.layers += [conv_relu_bn_drop(in_c, out_c, dilation=dilations[0], p=drop, norm_type='group', 
                                    actn='prelu', init_func=nn.init.kaiming_normal_)]
        
        for d,p,c in zip(dilations[:n], [drop]*n, [out_c]*n):
            self.layers += [conv_relu_bn_drop(c, c, dilation=d, p=drop, norm_type='group', 
                                    actn='prelu', init_func=nn.init.kaiming_normal_)]
            
        self.layers += [one_by_one_conv(out_c, num_classes)]
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x): return self.layers(x)
    
####################
    ## UNET ##
####################

class UnetBegin(nn.Module):
    'conv -> conv -> maxpool'
    def __init__(self, block_type, in_c, out_c, **kwargs):
        super(UnetBegin, self).__init__()
        self.conv_block1 = block_type(in_c, out_c, **kwargs)
        self.conv_block2 = block_type(out_c, out_c*2, **kwargs)
        self.pool = maxpool3D() 
        
    def forward(self, x):
        x = self.conv_block2(self.conv_block1(x))
        return self.pool(x), x
    
class UnetEnd(nn.Module):
    'conv -> conv -> maxpool 2**i + 2**(i+1), 2**i = in_c'
    def __init__(self, block_type, in_c=64, **kwargs):
        super(UnetEnd, self).__init__()
        i = int(math.log2(in_c))
        self.conv_block1 = block_type(2**i + 2**(i+1), 2**i, **kwargs)
        self.conv_block2 = block_type(2**i, 2**i, **kwargs)
        
    def forward(self, x):
        return self.conv_block2(self.conv_block1(x))

class UnetDownBlock(nn.Module):
    'conv -> conv -> maxpool'
    def __init__(self, block_type, in_c=64, **kwargs):
        super(UnetDownBlock, self).__init__()
        self.conv_block1 = block_type(in_c, in_c, **kwargs)
        self.conv_block2 = block_type(in_c, in_c*2, **kwargs)
        self.pool = maxpool3D() 
        
    def forward(self, x):
        x = self.conv_block2(self.conv_block1(x))
        return self.pool(x), x
    
class UnetUpBlock(nn.Module):
    'conv -> conv -> maxpool 2**i + 2**(i+1), 2**i = in_c'
    def __init__(self, block_type, in_c=64, **kwargs):
        super(UnetUpBlock, self).__init__()
        i = int(math.log2(in_c))
        self.conv_block1 = block_type(2**i + 2**(i+1), 2**i, **kwargs)
        self.conv_block2 = block_type(2**i, 2**i, **kwargs)
        
    def forward(self, x):
        x = self.conv_block2(self.conv_block1(x))
        return  F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
    
class VolumetricUnet(nn.Module):
    def __init__(self, in_c=1, out_c=32, n_layers=3, c=1, block_type=conv_relu_bn_drop, **kwargs):
        'create a 3d Unet with n_layers and out_c features'
        super(VolumetricUnet, self).__init__()

        # downblocks
        self.downblocks = nn.ModuleList([UnetBegin(block_type, in_c, out_c, **kwargs)] + 
                                        [UnetDownBlock(block_type, out_c*2**(i+1), **kwargs) for i in range(n_layers-1)])
        
        # middle
        self.middle = nn.Sequential(block_type(out_c*2**(n_layers), out_c*2**(n_layers), **kwargs),
                                    block_type(out_c*2**(n_layers), out_c*2**(n_layers+1), **kwargs))
        
        # upblocks
        self.upblocks = nn.ModuleList([UnetEnd(block_type, out_c*2, **kwargs)] +
                                      [UnetUpBlock(block_type, out_c*(2**(i+1)), **kwargs) for i in range(1, n_layers)])
        
        # final conv 1x1
        self.conv_final = one_by_one_conv(out_c*2, c)

    def forward(self, x):
        x_concats = []
        for l in self.downblocks:
            x, x_concat = l(x)
            x_concats += [x_concat]
        
        x = self.middle(x)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        
        for l, x_concat in zip(self.upblocks[::-1], x_concats[::-1]):
            x = l(torch.cat([x_concat, x], dim=1))
        
        return self.conv_final(x)
    

#########################
    ## RES-3D ##
#########################

def conv3d(ni:int, nf:int, ks:int=3, stride:int=1, pad:int=1, norm='batch'):
    bias = not norm == 'batch'
    conv = init_default(nn.Conv3d(ni,nf,ks,stride,pad,bias=bias))
    conv = spectral_norm(conv) if norm == 'spectral' else \
           weight_norm(conv) if norm == 'weight' else conv
    layers = [conv]
    layers += [nn.ReLU(inplace=True)]  # use inplace due to memory constraints
    layers += [nn.BatchNorm3d(nf)] if norm == 'batch' else []
    return nn.Sequential(*layers)
    
def res3d_block(ni, nf, ks=3, norm='batch', dense=False):
    """ 3d Resnet block of `nf` features """
    return SequentialEx(conv3d(ni, nf, ks, pad=ks//2, norm=norm),
                             conv3d(nf, nf, ks, pad=ks//2, norm=norm),
                             MergeLayer(dense))

def res3dmodel():
    norm = 'batch'
    layers = ([res3d_block(1,15,7,norm=norm,dense=True)] +
              [res3d_block(16,16,norm=norm) for _ in range(4)] +
              [conv3d(16,1,ks=1,pad=0,norm=None)])
    return nn.Sequential(*layers)

  
##################################
  ## Volumetric Residual Unet ##
##################################
    
def norm_act_conv_drop(in_channel, out_channel, dilation=1, p=0.5, norm_type='instance', actn='prelu', stride=1):
    'conv (pad=dilation) - same padding -> relu -> norm -> dropout'
    if norm_type == 'batch': norm = nn.BatchNorm3d(out_channel)
    if norm_type == 'instance': norm = nn.InstanceNorm3d(out_channel)
    if norm_type == 'group': norm = nn.GroupNorm(2, out_channel)

    if actn == 'relu': actn_fn = nn.ReLU(inplace=True)
    if actn == 'lrelu': actn_fn = nn.LeakyReLU(0.1)
    if actn == 'prelu': actn_fn = nn.PReLU()

    conv = nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=True)
    drop = nn.Dropout3d(p)
    return nn.Sequential(norm, actn_fn, conv, drop)

class PreActBlock(nn.Module):
    def __init__(self, in_c, **kwargs):
        super(PreActBlock, self).__init__()
        self.c1 = norm_act_conv_drop(in_c, in_c, **kwargs)
        self.c2 = norm_act_conv_drop(in_c, in_c, **kwargs)
        
    def forward(self, x):
        return x + self.c2(self.c1(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_c, **kwargs):
        super(UpsampleBlock, self).__init__()
        self.c = conv_relu_bn_drop(in_c, in_c//2, **kwargs)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        return self.c(x)

class LocalizationBlock(nn.Module):
    def __init__(self, in_c, **kwargs):
        super(LocalizationBlock, self).__init__()
        self.c1 = conv_relu_bn_drop(in_c, in_c, **kwargs)
        self.c2 = conv_relu_bn_drop(in_c, in_c//2, 1, **kwargs)
    
    def forward(self, x):
        return self.c2(self.c1(x))

class UpBlock(nn.Module):
    def __init__(self, in_c, **kwargs):
        super(UpBlock, self).__init__()    
        self.c1 = LocalizationBlock(in_c, **kwargs)
        self.c2 = UpsampleBlock(in_c//2, **kwargs)
        
    def forward(self, x):
        x1 = self.c1(x)
        return self.c2(x1), x1

class SegLayer(nn.Module):
    def __init__(self, in_c, **kwargs):
        super(SegLayer, self).__init__()
        self.c1 = conv_relu_bn_drop(in_c, in_c, 1, p=0.5, norm_type='instance', actn='prelu')
        self.c2 = one_by_one_conv(in_c, 1)
    
    def forward(self, x):
        return self.c2(self.c1(x))

class DownBlock(nn.Module):
    def __init__(self, in_c1, in_c2, first=False, **kwargs):
        super(DownBlock, self).__init__()
        if first: self.down = conv_relu_bn_drop(in_c1, in_c2, stride=1, **kwargs)
        else: self.down = conv_relu_bn_drop(in_c1, in_c2, stride=2, **kwargs)
        self.preact_res = PreActBlock(in_c2, **kwargs)
        
    def forward(self, x):
        return self.preact_res(self.down(x))    
 
    
class VolumetricResidualUnet(nn.Module):
    def __init__(self, in_c=8, **kwargs):
        super(VolumetricResidualUnet, self).__init__()
        
        # downblocks
        self.down1 = DownBlock(1, in_c, first=True, **kwargs)
        self.down2 = DownBlock(in_c, in_c*2**1, **kwargs)
        self.down3 = DownBlock(in_c*2**1, in_c*2**2, **kwargs)
        self.down4 = DownBlock(in_c*2**2, in_c*2**3, **kwargs)
    
        # middle
        self.middle = nn.Sequential(
            conv_relu_bn_drop(in_c*2**3, in_c*2**4, 1, stride=2, **kwargs),
            PreActBlock(in_c*2**4, **kwargs),
            UpsampleBlock(in_c*2**4, **kwargs)
        )

        # upblocks 
        self.upblock1 = UpBlock(in_c*2**4, **kwargs)
        self.upblock2 = UpBlock(in_c*2**3, **kwargs)
        self.upblock3 = UpBlock(in_c*2**2, **kwargs)

        # seg layers
        self.seg2 = SegLayer(in_c*2**2)
        self.seg3 = SegLayer(in_c*2**1)
        self.seg_final = SegLayer(in_c*2**1)
            
    def upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
    
    def forward(self, x):
        out1 = self.down1(x)
        out2 = self.down2(out1)
        out3 = self.down3(out2)
        out4 = self.down4(out3)
        middle_out = self.middle(out4)
        
        # concat1
        concat1 = torch.cat([out4, middle_out], dim=1)
        up1, _ = self.upblock1(concat1)
        
        # concat2
        concat2 = torch.cat([out3, up1], dim=1)
        up2, up2_segx = self.upblock2(concat2)
        
        # concat3
        concat3 = torch.cat([out2, up2], dim=1)
        up3, up3_segx = self.upblock3(concat3)
        
        # concat4
        concat4 = torch.cat([out1, up3], dim=1)
        
        # segmentation
        seg2out = self.seg2(up2_segx)
        seg3out = self.seg3(up3_segx)
        out_final = self.seg_final(concat4)
        return self.upsample(self.upsample(seg2out) + seg3out) + out_final
        
    
# #########################
#     ## MODIFIED UNET ##
# #########################

# class Modified3DUNet(nn.Module):
#     '''
#     - https://github.com/pykao/Modified-3D-UNet-Pytorch/blob/master/model.py
#     - https://www.med.upenn.edu/sbia/brats2017/rankings.html
#     '''
#     def __init__(self, in_channels, n_classes, base_n_filter = 8):
#         super(Modified3DUNet, self).__init__()
#         self.in_channels = in_channels
#         self.n_classes = n_classes
#         self.base_n_filter = base_n_filter

#         self.lrelu = nn.LeakyReLU()
#         self.dropout3d = nn.Dropout3d(p=0.6)
#         self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
#         #self.softmax = nn.Softmax(dim=1)

#         # Level 1 context pathway
#         self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
#         self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
#         self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

#         # Level 2 context pathway
#         self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False)
#         self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
#         self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter*2)

#         # Level 3 context pathway
#         self.conv3d_c3 = nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False)
#         self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
#         self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter*4)

#         # Level 4 context pathway
#         self.conv3d_c4 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
#         self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
#         self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter*8)

#         # Level 5 context pathway, level 0 localization pathway
#         self.conv3d_c5 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False)
#         self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16)
#         self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*8)

#         self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
#         self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

#         # Level 1 localization pathway
#         self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*16)
#         self.conv3d_l1 = nn.Conv3d(self.base_n_filter*16, self.base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
#         self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*4)

#         # Level 2 localization pathway
#         self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*8)
#         self.conv3d_l2 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*4, kernel_size=1, stride=1, padding=0, bias=False)
#         self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*2)

#         # Level 3 localization pathway
#         self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*4)
#         self.conv3d_l3 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*2, kernel_size=1, stride=1, padding=0, bias=False)
#         self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter)

#         # Level 4 localization pathway
#         self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter*2)
#         self.conv3d_l4 = nn.Conv3d(self.base_n_filter*2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)

#         self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter*8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
#         self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter*4, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)


#     def conv_norm_lrelu(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm3d(feat_out),
#             nn.LeakyReLU())

#     def norm_lrelu_conv(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.InstanceNorm3d(feat_in),
#             nn.LeakyReLU(),
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

#     def lrelu_conv(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.LeakyReLU(),
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

#     def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
#         return nn.Sequential(
#             nn.InstanceNorm3d(feat_in),
#             nn.LeakyReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             # should be feat_in*2 or feat_in
#             nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.InstanceNorm3d(feat_out),
#             nn.LeakyReLU())

#     def forward(self, x):
#         #  Level 1 context pathway
#         out = self.conv3d_c1_1(x)
#         residual_1 = out
#         out = self.lrelu(out)
#         out = self.conv3d_c1_2(out)
#         out = self.dropout3d(out)
#         out = self.lrelu_conv_c1(out)
#         # Element Wise Summation
#         out += residual_1
#         context_1 = self.lrelu(out)
#         out = self.inorm3d_c1(out)
#         out = self.lrelu(out)

#         # Level 2 context pathway
#         out = self.conv3d_c2(out)
#         residual_2 = out
#         out = self.norm_lrelu_conv_c2(out)
#         out = self.dropout3d(out)
#         out = self.norm_lrelu_conv_c2(out)
#         out += residual_2
#         out = self.inorm3d_c2(out)
#         out = self.lrelu(out)
#         context_2 = out

#         # Level 3 context pathway
#         out = self.conv3d_c3(out)
#         residual_3 = out
#         out = self.norm_lrelu_conv_c3(out)
#         out = self.dropout3d(out)
#         out = self.norm_lrelu_conv_c3(out)
#         out += residual_3
#         out = self.inorm3d_c3(out)
#         out = self.lrelu(out)
#         context_3 = out

#         # Level 4 context pathway
#         out = self.conv3d_c4(out)
#         residual_4 = out
#         out = self.norm_lrelu_conv_c4(out)
#         out = self.dropout3d(out)
#         out = self.norm_lrelu_conv_c4(out)
#         out += residual_4
#         out = self.inorm3d_c4(out)
#         out = self.lrelu(out)
#         context_4 = out

#         # Level 5
#         out = self.conv3d_c5(out)
#         residual_5 = out
#         out = self.norm_lrelu_conv_c5(out)
#         out = self.dropout3d(out)
#         out = self.norm_lrelu_conv_c5(out)
#         out += residual_5
#         out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

#         out = self.conv3d_l0(out)
#         out = self.inorm3d_l0(out)
#         out = self.lrelu(out)

#         # Level 1 localization pathway
#         out = torch.cat([out, context_4], dim=1)
#         out = self.conv_norm_lrelu_l1(out)
#         out = self.conv3d_l1(out)
#         out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

#         # Level 2 localization pathway
#         out = torch.cat([out, context_3], dim=1)
#         out = self.conv_norm_lrelu_l2(out)
#         ds2 = out
#         out = self.conv3d_l2(out)
#         out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

#         # Level 3 localization pathway
#         out = torch.cat([out, context_2], dim=1)
#         out = self.conv_norm_lrelu_l3(out)
#         ds3 = out
#         out = self.conv3d_l3(out)
#         out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

#         # Level 4 localization pathway
#         out = torch.cat([out, context_1], dim=1)
#         out = self.conv_norm_lrelu_l4(out)
#         out_pred = self.conv3d_l4(out)

#         ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
#         ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
#         ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
#         ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
#         ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

#         out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
#         return out
    



    

    
    
    
    