import torch.nn as nn
from torchvision.models import resnet34
from models.erp.erp_conv import ERPConv2d


def replace_layers(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            new_conv = ERPConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding[0],
                module.dilation,
                module.groups,
                module.bias is not None,
            )
            for target_param, param in zip(new_conv.parameters(), module.parameters()):
                target_param.data.copy_(param.data)
            # print('new', new_conv)
            # print('old', module)
            # print('------------------------------------')

            setattr(model, name, new_conv)
        else:
            replace_layers(module)


def erp_resnet_34(pretrained=True):
    net = resnet34(pretrained=pretrained)
    replace_layers(net)
    return net


if __name__ == "__main__":
    erp_resnet_34()
