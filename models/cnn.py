import torchvision.models as models
import torch.nn as nn
import torch
from torchsummary import summary
# feel free to add to the list: https://pytorch.org/docs/stable/torchvision/models.html
from datetime import datetime
import torch.nn.functional as F
### TWO HEAD MODELS ###


class TwoHeadResNet50Model(nn.Module):
    def __init__(self, hparams):
        super(TwoHeadResNet50Model, self).__init__()
        self.model = models.resnet50(pretrained=hparams.pretrained)
        # replace final layer with number of labels
        self.model.fc = Identity()
        self.fc_phase = nn.Linear(2048, 28)
        self.fc_tool = nn.Linear(2048, 20)

    def forward(self, x):
        now = datetime.now()
        out_stem = self.model(x)
        phase = self.fc_phase(out_stem)
        tool = self.fc_tool(out_stem)
        return out_stem, phase, tool

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        resnet50model_specific_args = parser.add_argument_group(
            title='resnet50model specific args options')
        resnet50model_specific_args.add_argument("--pretrained",
                                                 action="store_true",
                                                 help="pretrained on imagenet")
        resnet50model_specific_args.add_argument(
            "--model_specific_batch_size_max", type=int, default=80)
        return parser


#### Identity Layer ####
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class TwoHeadResNet50Model_tool(nn.Module):
    def __init__(self, hparams):
        super(TwoHeadResNet50Model_tool, self).__init__()
        self.opts = hparams
        self.model = models.resnet50(pretrained=True)
        # replace final layer with number of labels
        self.model.fc = Identity()

        self.embed_tool = nn.Linear(7, 64)

        self.fc_phase = nn.Linear(2048+64, 28)
        self.fc_tool = nn.Linear(2048+64, 20)

    def forward(self, x, tool_sig):
        # tool sig: T, 7

        # Tool Feature Embedding
        tool_fea = torch.tanh(self.embed_tool(tool_sig.float()))
        tool_fea = F.normalize(tool_fea, dim=-1, p=2)

        # ResNet Feature
        out_stem = self.model(x)

        out_stem = torch.cat([out_stem, tool_fea], 1)
        # out_stem = torch.tanh(self.embed_total(out_stem))
        # out_stem = F.normalize(out_stem, dim=-1, p=2)

        phase = self.fc_phase(out_stem)
        tool = self.fc_tool(out_stem)

        return out_stem, phase, tool

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        resnet50model_specific_args = parser.add_argument_group(
            title='resnet50model specific args options')
        resnet50model_specific_args.add_argument("--pretrained",
                                                 action="store_true",
                                                 help="pretrained on imagenet")
        resnet50model_specific_args.add_argument(
            "--model_specific_batch_size_max", type=int, default=80)
        return parser


class TwoHeadResNet50Model_phase(nn.Module):
    def __init__(self, hparams):
        super(TwoHeadResNet50Model_phase, self).__init__()
        self.opts = hparams
        self.model = models.resnet50(pretrained=True)
        # replace final layer with number of labels
        self.model.fc = Identity()

        self.embed_tool = nn.Linear(7, 64)

        self.fc_phase = nn.Linear(2048+64, 28)
        self.fc_tool = nn.Linear(2048+64, 20)

    def forward(self, x, phase_sig):
        # tool sig: T, 7

        # Tool Feature Embedding
        phase_fea = torch.tanh(self.embed_tool(phase_sig.float()))
        phase_fea = F.normalize(phase_fea, dim=-1, p=2)

        # ResNet Feature
        out_stem = self.model(x)

        out_stem = torch.cat([out_stem, phase_fea], 1)
        # out_stem = torch.tanh(self.embed_total(out_stem))
        # out_stem = F.normalize(out_stem, dim=-1, p=2)

        phase = self.fc_phase(out_stem)
        tool = self.fc_tool(out_stem)

        return out_stem, phase, tool

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        resnet50model_specific_args = parser.add_argument_group(
            title='resnet50model specific args options')
        resnet50model_specific_args.add_argument("--pretrained",
                                                 action="store_true",
                                                 help="pretrained on imagenet")
        resnet50model_specific_args.add_argument(
            "--model_specific_batch_size_max", type=int, default=80)
        return parser


class TwoHeadResNet50Model_multitask(nn.Module):
    def __init__(self, hparams):
        super(TwoHeadResNet50Model_multitask, self).__init__()
        self.opts = hparams
        self.model = models.resnet50(pretrained=True)
        # replace final layer with number of labels
        self.model.fc = Identity()

        self.embed_tool = nn.Linear(7, 64)
        self.embed_phase = nn.Linear(7, 64)

        self.fc_phase = nn.Linear(2048+64, 28)
        self.fc_tool = nn.Linear(2048+64, 20)

    def forward(self, x, tool_sig, phase_sig):
        # tool sig: T, 7

        # Tool Feature Embedding
        tool_fea = self.embed_tool(tool_sig.float())

        # Phase Feature Embedding
        phase_fea = self.embed_phase(phase_sig.float())

        # ResNet Feature
        out_stem = self.model(x)

        tool = self.fc_tool(torch.cat([out_stem, tool_fea], 1))
        phase = self.fc_phase(torch.cat([out_stem, phase_fea], 1))

        return out_stem, tool, phase

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        resnet50model_specific_args = parser.add_argument_group(
            title='resnet50model specific args options')
        resnet50model_specific_args.add_argument("--pretrained",
                                                 action="store_true",
                                                 help="pretrained on imagenet")
        resnet50model_specific_args.add_argument(
            "--model_specific_batch_size_max", type=int, default=80)
        return parser


######################## MSTCN-II ##############################
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

def make_layer(in_channel, out_channel, block_num, stride=1):
    shortcut = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, stride),
        nn.BatchNorm2d(out_channel))
    layers = list()
    layers.append(ResBlock(in_channel, out_channel, stride, shortcut))

    for i in range(1, block_num):
        layers.append(ResBlock(out_channel, out_channel))
    return nn.Sequential(*layers)


class semantic_encoding(nn.Module):
    def __init__(self, inchannel, hidden_state):
        super(semantic_encoding, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(inchannel, hidden_state, 7, 2, 3, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(True), nn.MaxPool2d(3, 2, 1))
        self.layer1 = make_layer(hidden_state, hidden_state, 2)

        self.avg = nn.AvgPool2d(56)

    def forward(self, x):
        '''

        :param x: one hot segmentation
        :param ini_mask: binary mask indicates the bounding box
        :return:
        '''
        # import torchvision.utils as utils
        # for i in range(7):
        #     utils.save_image(ini_mask[:, i:i + 1, ...], '../preprocess_tmp/binary' + str(i) + '.png')
        #     utils.save_image(x[:, i:i + 1, ...], '../preprocess_tmp/seg' + str(i) + '.png')
        # Apply Mask
        # if torch.max(ini_mask) == 0:
        #     # If no detection bounding box, all is zero
        #     ini_mask += 1
        # mask = torch.sigmoid(self.conv_mask2(F.leaky_relu(self.conv_mask1(ini_mask), 0.1, inplace=True)))

        # Semantic Map Encoding
        x = self.pre(x)
        x = self.layer1(x)

        # x = x * mask
        # x = x + res

        # Avg Pooling
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        return x

def exp_mask(val, mask):
    """Apply exponetial mask operation."""
    return torch.add(val, (1 - mask.type(torch.FloatTensor).cuda()) * -1e30)

class TwoHeadResNet50Model_multitask_ii(nn.Module):
    def __init__(self, hparams):
        super(TwoHeadResNet50Model_multitask_ii, self).__init__()
        self.opts = hparams
        self.model = models.resnet50(pretrained=True)
        # replace final layer with number of labels
        self.model.fc = Identity()

        self.embed_tool = nn.Linear(7, 64)
        self.embed_phase = nn.Linear(7, 64)

        self.sa_semantic = semantic_encoding(inchannel=7, hidden_state=64)

        self.embed_other_boxes_geo = nn.Linear(4, self.opts.box_emb_size)
        self.embed_other_boxes_class = nn.Linear(7, self.opts.box_emb_size)

        self.fc_phase = nn.Linear(2048+64+64+64+64, 28)
        self.fc_tool = nn.Linear(2048+64+64+64+64, 20)

    def forward(self, x, tool_sig, phase_sig, other_boxes, other_boxes_class, other_boxes_mask, one_hot_seg):
        # tool sig: T, 7

        # Tool Feature Embedding
        tool_fea = self.embed_tool(tool_sig.float())

        # Phase Feature Embedding
        phase_fea = self.embed_phase(phase_sig.float())

        # ResNet Feature
        out_stem = self.model(x)

        # Geo Feature
        # extract features from other boxes
        # obs_other_boxes [T, K, 4]
        # obs_other_boxes_class [T, K, num_class]
        # obs_other_boxes_mask [T, K]

        l, k, d1 = other_boxes.shape
        other_boxes = other_boxes.view(-1, d1)

        l, k, d2 = other_boxes_class.shape
        other_boxes_class = other_boxes_class.view(-1, d2)

        # [T, K, box_emb_size]
        obs_other_boxes_geo_features = torch.tanh(self.embed_other_boxes_geo(other_boxes))
        obs_other_boxes_geo_features = obs_other_boxes_geo_features.view(l, k, -1)

        obs_other_boxes_class_features = torch.tanh(self.embed_other_boxes_class(other_boxes_class))
        obs_other_boxes_class_features = obs_other_boxes_class_features.view(l, k, -1)

        obs_other_boxes_features = torch.cat([obs_other_boxes_geo_features, obs_other_boxes_class_features], 2)

        # cosine simi
        obs_other_boxes_geo_features = F.normalize(obs_other_boxes_geo_features, dim=-1, p=2)

        obs_other_boxes_class_features = F.normalize(obs_other_boxes_class_features, dim=-1, p=2)

        # [T, K]
        other_attention = torch.sum(torch.multiply(obs_other_boxes_geo_features, obs_other_boxes_class_features), 2)

        other_attention = exp_mask(other_attention, other_boxes_mask)

        other_attention = F.softmax(other_attention, dim=-1)

        # [T, K, 1] * [obs_len, K, feat_dim]
        # -> [obs_len, feat_dim]
        other_box_features_attended = torch.sum(other_attention.unsqueeze(-1) * obs_other_boxes_features, axis=1)

        # Semantic Feature
        attention_semantic = self.sa_semantic(one_hot_seg)

        tool = self.fc_tool(torch.cat([out_stem, tool_fea, other_box_features_attended, attention_semantic], 1))
        phase = self.fc_phase(torch.cat([out_stem, phase_fea, other_box_features_attended, attention_semantic], 1))

        out_stem_tool = torch.cat([out_stem, tool_fea, other_box_features_attended, attention_semantic], 1)
        out_stem_phase = torch.cat([out_stem, phase_fea, other_box_features_attended, attention_semantic], 1)

        return out_stem_tool, out_stem_phase, tool, phase

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        resnet50model_specific_args = parser.add_argument_group(
            title='resnet50model specific args options')
        resnet50model_specific_args.add_argument("--pretrained",
                                                 action="store_true",
                                                 help="pretrained on imagenet")
        resnet50model_specific_args.add_argument(
            "--model_specific_batch_size_max", type=int, default=80)
        return parser