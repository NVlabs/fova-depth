import torch
import torch.nn as nn
import torch.nn.functional as F
from models.erp.erp_conv import ERPConv2d
from models.erp.resnet_erp import erp_resnet_34

class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """
    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 =  ERPConv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 =  ERPConv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.relu = nn.ReLU(inplace=True)
       

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x




class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit = ResidualConvUnit(features)

    def forward(self,x, res=None ):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = x

        if res is not None:
            output = output + self.resConfUnit(res)

        output = self.resConfUnit(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear")
        return output


class MidasERPFeatureExtractor(nn.Module):
    def __init__(self, pretrained=False, input_ch=3):
        super().__init__()
        resnet = erp_resnet_34(pretrained=pretrained)
       
        if input_ch == 3:
            self.resnet_conv1 = resnet.conv1
        else:
            self.resnet_conv1 = ERPConv2d(input_ch, 64, 7, stride=2, bias=False, padding=3)


        self.resnet_bn1 = resnet.bn1
        self.resnet_relu = resnet.relu
        self.resnet_maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3 
        self.layer4 = resnet.layer4
      
 
    
    def forward(self, erps):
     
        x = self.resnet_conv1( erps)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
       
        return layer1, layer2, layer3, layer4




class OutputConv(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = ERPConv2d(features, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = ERPConv2d(128, 1, kernel_size=3, stride=1, padding=1)
       

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.functional.interpolate(x,scale_factor=2,mode='bilinear')
        return x

class OutputConvNew(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = ERPConv2d(features, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = ERPConv2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = ERPConv2d(32,1, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.interpolate(x,scale_factor=2,mode='bilinear')
        x = self.conv3(x)
        return x


        

class ERPMidasNet(nn.Module):
   
    def __init__(self, features=256, out_ch=1, pretrained=True, input_ch=3, model='resnet34', use_new_output=False):
        super().__init__()

        if model == 'resnet34':
            layer_out_channels=[64,128,256,512]
        else:
            raise RuntimeError('Cube Midas feature extractor resnet name not recognized', model)
        

        self.pretrained = MidasERPFeatureExtractor(pretrained=pretrained, input_ch=input_ch)

        self.refinenet4 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet1 = FeatureFusionBlock(features)

        self.layer1_rn = ERPConv2d(layer_out_channels[0], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = ERPConv2d(layer_out_channels[1], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = ERPConv2d(layer_out_channels[2], features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = ERPConv2d(layer_out_channels[3], features, kernel_size=3, stride=1, padding=1, bias=False)

        if use_new_output:
            self.output_conv = OutputConvNew(features)
        else:
            self.output_conv = OutputConv(features)


    def forward(self, erps):
       
        layer_1, layer_2, layer_3, layer_4 = self.pretrained(erps)
        
        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, res=layer_3_rn)
        path_2 = self.refinenet2(path_3, res=layer_2_rn)
        path_1 = self.refinenet1(path_2, res=layer_1_rn)

        out = self.output_conv(path_1)
        return out