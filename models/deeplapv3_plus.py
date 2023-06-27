import torch
import torch.nn as nn
import torch.nn.functional as F
# mobilenetv2网络下方已给出
from models.mobilenetv2 import mobilenetv2


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        # ---------------------------------------------------------#
        #   把最后一层卷积剔除，也就是
        #   17 InvertedResidual后跟着的 18 常规1x1卷积 剔除
        # ---------------------------------------------------------#
        self.features = model.features[:-1]
        # ----------------------------------------------------------------------#
        #   18 = 开始的常规conv + 17 个InvertedResidual，即features.0到features.17
        # ----------------------------------------------------------------------#
        self.total_idx = len(self.features)
        # ---------------------------------------------------------#
        #   每个 下采样block 所处的索引位置
        #   即Output Shape h、w尺寸变为原来的1/2
        # ---------------------------------------------------------#
        self.down_idx = [2, 4, 7, 14]

        # -------------------------------------------------------------------------------------------------#
        #   若下采样倍数为8，则网络会进行3次下采样(features.0，features.2，features.4)，尺寸 512->64
        #       需要对后两处下采样block(步长s为2的InInvertedResidual)的参数进行修改，使其变为空洞卷积,尺寸不再下降
        #       再解释一下，下采样倍数为8，表示输入尺寸缩小为原来的1/8，也就是经历3次步长为2的卷积
        #
        #   若下采样倍数为16，则会进行4次下采样(features.0，features.2，features.4，features.7)，尺寸 512-> 32
        #       只需要对最后一处 下采样block 的参数进行修改
        # -------------------------------------------------------------------------------------------------#
        if downsample_factor == 8:
            # ----------------------------------------------#
            #   从第features.7到features.13
            # ----------------------------------------------#
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                # ----------------------------------------------#
                #   apply(func,...)：func参数是函数，相当于C/C++的函数指针。
                #   partial函数用于携带部分参数生成一个新函数
                # ----------------------------------------------#
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            # ----------------------------------------------#
            #   从第features.14到features.17
            # ----------------------------------------------#
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    # ----------------------------------------------------------------------#
    #   _nostride_dilate函数目的：通过修改卷积参数实现 self.features[i] 尺寸不变
    # ----------------------------------------------------------------------#
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):  # 原本步长为2的
                m.stride = (1, 1)  # 步长变为1
                if m.kernel_size == (3, 3):  # kernel_size为3的
                    m.dilation = (dilate // 2, dilate // 2)  # 膨胀系数变为dilate参数的一半
                    m.padding = (dilate // 2, dilate // 2)  # 填充系数变为dilate参数的一半
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        # ------------------------------------------------------------------------------#
        #   low_level_features表示低(浅)层语义特征，只进行了features.0和features.2两次下采样，
        #       features.3的输出尺寸和features.2一样
        #       输入为512x512，下采样倍数为16时，CHW：[24, 128, 128]
        # ------------------------------------------------------------------------------#
        low_level_features = self.features[:4](x)
        # ------------------------------------------------------#
        #   x表示高(深)层语义特征，其h、w尺寸更小些
        #       输入为512x512，下采样倍数为16时，CHW：[320, 32, 32]
        # ------------------------------------------------------#
        x = self.features[4:](low_level_features)
        return low_level_features, x

    # -----------------------------------------#


#   ASPP特征提取模块
#   得到深层特征后，进行加强特征提取
#   利用 不同膨胀率rate 的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # -----------------------------------------#
        #   结合forward中第五个分支去看
        # -----------------------------------------#
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        # -----------------------------------------#
        #   五个分支堆叠后的特征，经1x1卷积去整合特征
        # -----------------------------------------#
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        # ---------------------------------------------#
        #   利用插值方法，对输入的张量数组进行上\下采样操作
        #       这样才能去和上面四个特征图进行堆叠
        #       (row, col)：输出空间的大小
        # ---------------------------------------------#
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result  # 图中Encoder部分，1x1 Conv后的绿色特征图


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [32,32,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320  # backbone深层特征引出来的通道数
            low_level_channels = 24  # backbone浅层特征引出来的通道数
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        # -----------------------------------------#
        #   ASPP特征提取模块(加强特征提取)
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #   得到Encoder部分，1x1 Conv后的绿色特征图
        # -----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        # ----------------------------------#
        #   浅层特征边
        #   Decoder部分1x1卷积进行通道数调整
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # ----------------------------------#
        #   Decoder部分，对堆叠后的特征图进行
        #       两次3x3的特征提取
        # ----------------------------------#
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        # --------------------------------------------------#
        #   输入图片的高和宽，最后上采样得到的输出层和此保持一致
        # --------------------------------------------------#
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))  # 堆叠 + 3x3卷积特征提取
        # -----------------------------------------#
        #   对获取到的特征进行分类，获取每个像素点的种类
        #   对于VOC数据集，输出尺寸CHW为[21, 128, 128]
        #   21个类别，这儿就输出21个channel，
        #	然后经过softmax以及argmax等操作完成像素级分类任务
        # -----------------------------------------#
        x = self.cls_conv(x)
        # -----------------------------------------#
        #   通过上采样使得最终输出层，高宽和输入图片一样。
        # -----------------------------------------#
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


if __name__ == "__main__":
    num_classes = 21  # 语义分割，VOC数据集,21个类别
    model = DeepLab(num_classes, backbone="mobilenet", pretrained=False, downsample_factor=16)
    model.eval()
    print(model)

    # --------------------------------------------------#
    #   用来测试网络能否跑通，同时可查看FLOPs和params
    # --------------------------------------------------#
    # from torchsummaryX import summary
    #
    # summary(model, torch.randn(1, 3, 512, 512))
