from decimal import Decimal


def txtTransToDict(fileName):
    File = open(fileName, 'r')
    Dic = {}
    keys = []  # 用来存储读取的顺序
    for line in File:
        value = line.strip().split(' ')
        Dic[value[0]] = value[1]
        keys.append(value[0])
    File.close()
    return Dic


"""
predict_coordinate.txt
阈值为1%时，准确率： 0.42857142857142855
阈值为3%时，准确率： 0.8721804511278195
阈值为5%时，准确率： 0.9924812030075187
阈值为10%时，准确率： 1.0
阈值为15%时，准确率： 1.0
平均引用误差： 0.01444347981061114586466165414

predict_SE_mid.txt
阈值为1%时，准确率： 0.41353383458646614
阈值为3%时，准确率： 0.8947368421052632
阈值为5%时，准确率： 0.9849624060150376
阈值为10%时，准确率： 1.0
阈值为15%时，准确率： 1.0
平均引用误差： 0.01508723685981683413533834586
"""

def verifivate():
    predictDic = txtTransToDict("result/predict-net.txt")
    # predictDic = txtTransToDict("result/predict_SE_mid.txt")
    # predictDic = txtTransToDict("result/predict_pspnet.txt")
    # predictDic = txtTransToDict("result/predict-net_improve.txt")
    # predictDic = txtTransToDict("result/predict_co_encode.txt")
    labelDic = txtTransToDict("label.txt")
    verificateFlag = 0  # 验证成功为0，失败为1
    keysOfPredict = predictDic.keys()
    # print(keysOfPredict)
    # print(f"数据长度为:{keysOfPredict.__len__()}")
    accuray1 = 0
    accuray3 = 0
    accuray5 = 0
    accuray10 = 0
    accuray15 = 0
    length = keysOfPredict.__len__()
    iterNum = 0  # 迭代次数
    sumloss = 0
    for key in keysOfPredict:
        currentValueOfPredict = predictDic.get(key)  # predict当前图片的数值
        valueOfLabel = labelDic.get(key)  # label当前图片的数值
        # print(currentValueOfPredict,valueOfLabel)
        loss = abs(Decimal(currentValueOfPredict) - Decimal(valueOfLabel))
        # print(loss)
        if loss <= 0.005:
            accuray1 +=1
        if loss<=0.01:
            accuray3 +=1
            # print(key)
        if loss<=0.02:
            accuray5 +=1
            # print(key)
        if loss <= 0.05:
            accuray10 +=1
        if loss <=0.1:
            accuray15 +=1
        sumloss =sumloss + loss
    lastacc1 = accuray1/length
    lastacc3 = accuray3 / length
    lastacc5 = accuray5 / length
    lastacc10 = accuray10 / length
    lastacc15 = accuray15 / length
    print("阈值为0.5%时，准确率：",lastacc1)
    print("阈值为1%时，准确率：", lastacc3)
    print("阈值为2%时，准确率：", lastacc5)
    print("阈值为5%时，准确率：", lastacc10)
    print("阈值为10%时，准确率：", lastacc15)
    print("平均引用误差：",sumloss/134)

if __name__ == '__main__':
    verifivate()
