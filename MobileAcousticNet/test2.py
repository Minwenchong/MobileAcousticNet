import numpy
import torch

from Models.AcousticNet3 import device
from GetRealWorldData import test_dl
import torch.nn.functional as F

Y_Pred = []
Y_label = []
def inference(model, val_dl):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)
            # outputs = F.softmax(outputs, dim=1)
            outputs = F.sigmoid(outputs)
            # print(outputs)
            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)

            print("this is label", labels)
            Y_label.append(labels.cpu())
            print("this is pred ", prediction)
            Y_Pred.append(prediction.cpu())
            print('*************************************')

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            print(correct_prediction)
            total_prediction += prediction.shape[0]
    acc = correct_prediction / total_prediction
    print(f'Accuracy: {acc:.5f}, Total items: {total_prediction}')



myModel = torch.load("model.pt")
# 参数量过大的主要原因是有太多的全连接层用来计算权重了
print("mode is load!")
# from thop import profile
# from thop import clever_format
# input=torch.randn(1,2,13,403).cuda()
# flops, params = profile(myModel, inputs=(input, ))
# print(flops, params)
# flops, params = clever_format([flops, params], "%.3f")
# print(flops, params)
inference(myModel,test_dl)
import numpy as np
for i in range(len(Y_Pred)):
    Y_Pred[i] = np.array(Y_Pred[i])
print()
for i in range(len(Y_label)):
    Y_label[i] = np.array(Y_label[i])

import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
Y_Pred = np.reshape(Y_Pred,newshape=(24*16))
Y_label = np.reshape(Y_label,newshape=(24*16))
print(Y_Pred.shape)
print(Y_label.shape)
cm = confusion_matrix(Y_label, Y_Pred)
print(cm)
labels = ['clean','infested']
# 转换成dataframe，转不转一样
df_cm = pd.DataFrame(cm,index=labels,columns=labels)

# annot = True 显示数字 ，fmt参数不使用科学计数法进行显示
ax = sn.heatmap(df_cm, annot=True, fmt='.20g',cmap='GnBu',
                annot_kws={'size':12,'weight':'bold', 'color':'black'})
# ax.set_title('confusion matrix')  # 标题
ax.set_xlabel('Predict label')  # x轴
ax.set_ylabel('True label')  # y轴
plt.show()
