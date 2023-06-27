import math
import os
import cv2
import numpy
import torch
from models.u2net_CBAM import U2NET
#from VerificateTxtError import *

class MeterReader(object):

    def __init__(self, is_cuda=False):
        self.net = U2NET(3, 2)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and is_cuda else 'cpu')
        self.net.load_state_dict(torch.load('weight/u2net_CBAM.pt', map_location='cpu'))
        self.net.eval().to(self.device)

        '''以下为超参数，需要根据不同表盘类型设定'''
        self.line_width = 1600  # 表盘展开为直线的长度,按照表盘的图像像素周长计算
        self.line_height = 150  # 表盘展开为直线的宽度，该设定按照围绕圆心的扇形宽度计算，需要保证包含刻度以及一部分指针
        self.circle_radius = 200  # 预设圆盘直径，扫描的最大直径，需要小于图幅，否者可能会报错
        self.circle_center = [208, 208]  # 圆盘指针的旋转中心，预设的指针旋转中心
        self.pi = 3.1415926535898  # 参数PI
        self.threshold = 0.5  # 分割的阈值

    @torch.no_grad()
    def __call__(self, image):
        image = self.square_picture(image, 416)#图像正方形化，边缘背景色填充
        image_tensor = self.to_tensor(image.copy()).to(self.device)#对图像通道顺序变换，归一并添加新维度
        d0, d1, d2, d3, d4, d5, d6 = self.net(image_tensor)
        mask = d0.squeeze(0).cpu().numpy()#去掉第一维度 mask是整体刻度和指针图
        point_mask = self.binary_image(mask[0])#指针二值化[416,416]
        dail_mask = self.binary_image(mask[1])#表盘二值化[416,416]
        relative_value = self.get_relative_value(point_mask, dail_mask)
        # print(relative_value)

        """绘制分割结果图"""
        # cv2.imshow('point_mask', point_mask)
        # cv2.imshow('dail_mask', dail_mask)
        # cv2.imshow('image', image)
        condition = point_mask ==1
        image[condition] = (0,0,255)
        condition = dail_mask == 1
        image[condition] = (0, 255, 0)
        #输出
        # cv2.imshow('image_mask', image)
        cv2.waitKey()

        # return relative_value['ratio']
        return relative_value
    def binary_image(self, image):
        """图片二值化"""
        condition = image > self.threshold#
        image[condition] = 1
        image[~condition] = 0
        image = self.corrosion(image)
        return image
    #获取指针和刻度的相对值
    def get_relative_value(self, image_pointer, image_dail):
        import matplotlib.pyplot as plt
        '''表盘图片展平'''
        line_image_pointer = self.create_line_image(image_pointer)
        line_image_dail = self.create_line_image(image_dail)
        '''二维图像转换为1为数组'''
        data_1d_pointer = self.convert_1d_data(line_image_pointer)#shape是1600 指针
        data_1d_dail = self.convert_1d_data(line_image_dail)#刻度
        data_1d_dail = self.mean_filtration(data_1d_dail)  # 均值滤波

        """绘制一维数组结果图"""
        # plt.plot(numpy.arange(0, len(data_1d_pointer)), data_1d_pointer)
        # plt.plot(numpy.arange(0, len(data_1d_dail)), data_1d_dail)
        # plt.show()
        # exit()
        # cv2.imshow('line_image_pointer', line_image_pointer)
        # cv2.imshow('line_image_dail', line_image_dail)
        '''定位指针相对刻度位置'''
        dail_flag = False
        pointer_flag = False
        one_dail_start = 0
        one_dail_end = 0
        one_pointer_start = 0
        one_pointer_end = 0
        dail_location = []
        pointer_location = 0
        for i in range(self.line_width - 1):
            #该阶段处理刻度
            if data_1d_dail[i] > 0 and data_1d_dail[i + 1] > 0:#检测到刻度值
                if not dail_flag:
                    one_dail_start = i#标记刻度的起始位置
                    dail_flag = True
            if dail_flag:
                if data_1d_dail[i] == 0 and data_1d_dail[i + 1] == 0:#检测到刻度的边缘，也就是刻度末尾为空，此时整个刻度被读取进来
                    one_dail_end = i - 1#刻度最后边缘位置
                    one_dail_location = (one_dail_start + one_dail_end) / 2#计算刻度的中心位置
                    dail_location.append(one_dail_location)#添加刻度标志
                    one_dail_start = 0#重置
                    one_dail_end = 0
                    dail_flag = False
            #该阶段处理指针
            if data_1d_pointer[i] > 0 and data_1d_dail[i + 1] > 0:#指针起点开始
                if not pointer_flag:
                    one_pointer_start = i#记录指针初始位置
                    pointer_flag = True
            if pointer_flag:
                if data_1d_pointer[i] == 0 and data_1d_pointer[i + 1] == 0:#检测指针的结束位置
                    one_pointer_end = i - 1
                    pointer_location = (one_pointer_start + one_pointer_end) / 2
                    one_pointer_start = 0
                    one_pointer_end = 0
                    pointer_flag = False
        scale_num = len(dail_location)#记录总刻度数
        num_scale = -1
        ratio = -1
        if scale_num > 0:
            for i in range(scale_num - 1):
                if dail_location[i] <= pointer_location < dail_location[i + 1]:#指针夹在两个刻度之间的情况
                    num_scale = i + (pointer_location - dail_location[i]) / (
                            dail_location[i + 1] - dail_location[i] + 1e-5) + 1#i从0开始 需要补1
            #ratio = (pointer_location - dail_location[0]) / (dail_location[-1] - dail_location[0] + 1e-5)
            ratio = num_scale / scale_num
        result = {'scale_num': scale_num, 'num_sacle': num_scale, 'ratio': ratio}
        # if scale_num < 45:
        #     print("推测刻度为:",round(ratio*1.6,2))
        # else:
        #     print("推测刻度为:",round(ratio*25,2))
        # print("推测刻度比例ratio为:",ratio)
        print(result)
        return result

    def create_line_image(self, image_mask):
        import matplotlib.pyplot as plt
        """
        把表盘图像水平展开
        创建线性图
        :param image_mask: 掩码图
        :return:
        """
        line_image = numpy.zeros((self.line_height, self.line_width), dtype=numpy.uint8)#[150,1600]h,w
        for row in range(self.line_height):
            for col in range(self.line_width):
                """计算与-y轴的夹角"""
                theta = ((2 * self.pi) / self.line_width) * (col + 1)#占总宽度的百分比*2pi
                '''计算当前扫描点位对应于原图的半径'''
                radius = self.circle_radius - row - 1
                '''计算当前扫描点对应于原图的位置'''
                y = int(self.circle_center[0] + radius * math.cos(theta) + 0.5)
                x = int(self.circle_center[1] - radius * math.sin(theta) + 0.5)
                # y = int(self.circle_center[0] + radius * math.cos(theta))
                # x = int(self.circle_center[1] - radius * math.sin(theta))
                line_image[row, col] = image_mask[y, x]
        # plt.imshow(line_image)
        # plt.show()
        return line_image

    def convert_1d_data(self, line_image):
        """
        将图片转换为1维数组
        :param line_image: 展开的图片
        :return: 一维数组
        """
        data_1d = numpy.zeros((self.line_width), dtype=numpy.int16)
        for col in range(self.line_width):
            for row in range(self.line_height):
                if line_image[row, col] == 1:
                    data_1d[col] += 1
        return data_1d
    #图像进行腐蚀处理后，高亮区减少
    def corrosion(self, image):
        """
        腐蚀操作
        :param image:
        :return:
        """
        kernel = numpy.ones((3, 3), numpy.uint8)
        image = cv2.erode(image, kernel)
        return image

    def mean_filtration(self, data_1d_dail):
        """
        均值滤波
        :param data_1d_dail:
        :return:
        """
        mean_data = numpy.mean(data_1d_dail)
        for col in range(self.line_width):
            if data_1d_dail[col] < mean_data:
                data_1d_dail[col] = 0
        return data_1d_dail

    @staticmethod
    #torch读取顺序chw opencv读取顺序hwc tensorflow读取顺序hwc
    def to_tensor(image):
        image = torch.tensor(image).float() / 255
        image = image.permute(2, 0, 1).unsqueeze(0)#转换通道顺序并添加新维度 添加个batch size维度
        return image

    @staticmethod
    def square_picture(image, image_size):
        """
        任意图片正方形中心化
        :param image: 图片
        :param image_size: 输出图片的尺寸
        :return: 输出图片
        """
        h1, w1, _ = image.shape
        max_len = max(h1, w1)
        fx = image_size / max_len
        fy = image_size / max_len
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)#放大图像
        h2, w2, _ = image.shape#获取放大图像后的大小
        background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
        background[:, :, :] = 127#设置背景颜色进行填充
        s_h = image_size // 2 - h2 // 2
        s_w = image_size // 2 - w2 // 2
        background[s_h:s_h + h2, s_w:s_w + w2] = image#把图像放到背景上
        return background


if __name__ == '__main__':
    tester = MeterReader()
    root = 'data/newdata'
    for image_name in os.listdir(root):
        path = f'{root}/{image_name}'
        image = cv2.imread(path)#cv读取是hwc 需要转换成chw
        print(image_name, end=" ")
        result = tester(image)
        #第一次推理写入文件
        with open('result/predict_CBAM.txt', 'a') as file:
            ratio = result['ratio']
            file.write((f'{image_name} {ratio}\n'))
    #写入后可检查正确率
    # verifivate()
        # cv2.imshow('a', image)
        # cv2.waitKey()
