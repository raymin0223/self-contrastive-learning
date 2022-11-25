import re
import math
import collections
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


def get_width_and_height_from_size(x):
    """ Obtains width and height from a int or tuple """
    if isinstance(x, int): return x, x
    if isinstance(x, list) or isinstance(x, tuple): return x
    else: raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """ Calculates the output image size when using Conv2dSamePadding with a stride. 
        Necessary for static padding. Thanks to mannatsingh for pointing this out. """
    if input_image_size is None: return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input
    

class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """
    
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.
        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args
    
    
class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params, image_size=None, drop_connect_rate=0.2):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect
        self.drop_connect_rate = drop_connect_rate

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this would do nothing
        
        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1,1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = Swish()

    def forward(self, inputs):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """
        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if self.drop_connect_rate:
                x = drop_connect(x, p=self.drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, selfcon_pos=[False]):
        super().__init__()
        blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
        ]

        blocks_args[1] = 'r2_k3_s11_e6_i16_o24_se0.25'
        blocks_args = BlockDecoder.decode(blocks_args)
        
        params = {'b0': (1.0, 1.0, 32, 0.2), 'b1': (1.0, 1.1, 34, 0.2), 'b2': (1.1, 1.2, 38, 0.3)}
        w, d, s, p = params['b0']
        
        global_params = GlobalParams(
            batch_norm_momentum=0.99,
            batch_norm_epsilon=1e-3,
            dropout_rate=p,
            drop_connect_rate=0.2,
            width_coefficient=w,
            depth_coefficient=d,
            depth_divisor=8,
            min_depth=None,
            image_size=s,
        )
    
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        stride = 1
            
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        block_layers = []
        drop_connect_rate = self._global_params.drop_connect_rate
        num = 0
        for b in self._blocks_args:
            num += b.num_repeat
            
        index = 0
        for block_args in self._blocks_args:
            layers = []
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            layers.append(MBConvBlock(block_args, self._global_params, image_size=image_size, drop_connect_rate=drop_connect_rate*index/num))
            index += 1
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                layers.append(MBConvBlock(block_args, self._global_params, image_size=image_size, drop_connect_rate=drop_connect_rate*index/num))
                index += 1
                
            block_layers.append(nn.Sequential(*layers))
        self.block_layers= nn.ModuleList(block_layers)

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(512, self._global_params)
        self.final_channels = out_channels

        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._swish = Swish()

        sub_conv = []
        sub_conv.append(Conv2d(80, self.final_channels, kernel_size=1, bias=False))
        sub_conv.append(nn.BatchNorm2d(num_features=self.final_channels, momentum=bn_mom, eps=bn_eps))
        sub_conv.append(Swish())

        self.selfcon_layer = self._make_sub_layer(selfcon_pos, nn.Sequential(*sub_conv))
        
    # simply test with nn.Linear
    def _make_sub_layer(self, pos, sub_conv):
        pos = pos[0]
        if not pos:
            return None
        else:
            return nn.ModuleList([sub_conv, nn.Linear(self.final_channels, self.final_channels)])

    # Stem
    def conv_stem(self, x):
        x = self._swish(self._bn0(self._conv_stem(x)))
        
        return x
    
    def pool_linear(self, feat):
        # Head
        feat = self._swish(self._bn1(self._conv_head(feat))) 

        # Pooling and final linear layer
        feat = self._avg_pooling(feat)
        features = feat.view(feat.size(0), -1)
        features = self._dropout(features)

        return features
        
    def forward(self, x):
        sub_out = []

        x = self.conv_stem(x)
        
        for i in range(4):
            x = self.block_layers[i](x)
        
        if self.selfcon_layer is not None:
            out = self.selfcon_layer[0](x)
            out = torch.flatten(self._avg_pooling(out), 1)
            out = self._dropout(out)
            out = self.selfcon_layer[1](out)
            sub_out.append(out)

        for i in range(4, len(self.block_layers)):
            x = self.block_layers[i](x)

        features = self.pool_linear(x)

        return sub_out, features
    

def efficientnet(**kwargs):
    return EfficientNet(**kwargs)


model_dict = {
    'efficientnet': [efficientnet, 512]
}


class ConEfficientNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='efficientnet', head='mlp', feat_dim=128, selfcon_pos=[False,False,False], selfcon_arch='resnet', selfcon_size='same', dataset=''):
        super(ConEfficientNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(selfcon_pos=selfcon_pos)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
            
            self.sub_heads = []
            for pos in selfcon_pos:
                if pos:
                    self.sub_heads.append(nn.Linear(dim_in, feat_dim))
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
            
            heads = []
            for pos in selfcon_pos:
                if pos:
                    heads.append(nn.Sequential(
                        nn.Linear(dim_in, dim_in),
                        nn.ReLU(inplace=True),
                        nn.Linear(dim_in, feat_dim)
                    ))
            self.sub_heads = nn.ModuleList(heads)
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        sub_feat, feat = self.encoder(x)
        
        sh_feat = []
        for sf, sub_head in zip(sub_feat, self.sub_heads):
            sh_feat.append(F.normalize(sub_head(sf), dim=1))
        
        feat = F.normalize(self.head(feat), dim=1)
        return sh_feat, feat


class CEEffNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='efficientnet', method='ce', num_classes=10, dim_out=128, selfcon_pos=[False], selfcon_arch='resnet', selfcon_size='same', dataset=''):
        super(CEEffNet, self).__init__()
        self.method = method
        
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(selfcon_pos=selfcon_pos)
        
        logit_fcs, feat_fcs = [], []
        for pos in selfcon_pos:
            if pos:
                logit_fcs.append(nn.Linear(dim_in, num_classes))
                feat_fcs.append(nn.Linear(dim_in, dim_out))
                
        self.logit_fc = nn.ModuleList(logit_fcs)
        self.l_fc = nn.Linear(dim_in, num_classes)
        
        if method not in ['ce', 'subnet_ce', 'kd']:
            self.feat_fc = nn.ModuleList(feat_fcs)
            self.f_fc = nn.Linear(dim_in, dim_out)
            for param in self.f_fc.parameters():
                param.requires_grad = False

    def forward(self, x):
        sub_feat, feat = self.encoder(x)
        
        feats, logits = [], []
        
        for idx, sh_feat in enumerate(sub_feat):
            logits.append(self.logit_fc[idx](sh_feat))
            if self.method not in ['ce', 'subnet_ce', 'kd']:
                out = self.feat_fc[idx](sh_feat)
                feats.append(F.normalize(out, dim=1))
            
        if self.method not in ['ce', 'subnet_ce', 'kd']:
            return [feats, F.normalize(self.f_fc(feat), dim=1)], [logits, self.l_fc(feat)]
        else:
            return [logits, self.l_fc(feat)]


class LinearClassifier_EFF(nn.Module):
    """Linear classifier"""
    def __init__(self, name='efficientnet', num_classes=100):
        super(LinearClassifier_EFF, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)