
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from Models.MobileNetV3 import myModel, device
from GetRealWorldData import train_dl, val_dl
import torch.nn.functional as F
#### 目前虽然极大的缩小了模型的大小，但是模型的精度仍然需要提升，该怎么办呢！！！！！

from EarlyStop import EarlyStopping

save_path = ".\\"  # 当前目录下
early_stopping = EarlyStopping(save_path)

train_loss = []
train_acc = []
valid_acc = []
valid_loss = []

def training(model, train_dl, num_epochs):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        val_loss = 0.0

        model.train()
        for i, data in enumerate(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = F.sigmoid(outputs)
            output_loss = outputs[:, 1]
            loss = criterion(output_loss, labels.float())
            # loss = loss + model.reg_loss(alpha=0.03)# 这个alpha的值怎么设置？
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        train_loss.append(avg_loss)
        train_acc.append(acc)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.8f}, Accuracy: {acc:.2f}')


        for i, data in enumerate(val_dl):
            inputs, labels = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            outputs = model(inputs)
            outputs = F.sigmoid(outputs)
            output_loss = outputs[:, 1]
            loss = criterion(output_loss, labels.float())
            # loss = loss + model.reg_loss(alpha=0.03)
            val_loss += loss.item()

            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        num_batches = len(val_dl)
        acc = correct_prediction / total_prediction
        avg_loss = val_loss / num_batches
        valid_loss.append(avg_loss)
        valid_acc.append(acc)
        print(f'Epoch: {epoch}, Loss_val: {avg_loss:}, Accuracy: {acc:}')

        early_stopping(avg_loss, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break  # 跳出迭代，结束训练

    print('Finished Training')

def paint_training_acc(val,train):
    x = range(0,100)
    plt.figure(figsize=(15,4))
    plt.xticks(np.arange(0,100,5))
    plt.plot(x,val,label="valSet",linewidth=1.5,color='b')
    plt.plot(x,train,label="trainSet",linewidth=1.5,color='g')
    plt.xlabel("epoch")
    # plt.ylabel("val/train")
    plt.legend(loc="upper right")
    plt.savefig("real_acc")
    plt.show()
def paint_training_loss(val,train):
    x = range(0,100)
    plt.figure(figsize=(15,4))
    plt.xticks(np.arange(0,100,5))
    plt.plot(x,val,label="valset",linewidth=1.5,color='b')
    plt.plot(x,train,label="trainSet",linewidth=1.5,color='g')
    plt.xlabel("epoch")
    # plt.ylabel("val/train")
    plt.legend(loc="upper right")
    plt.savefig("real_loss")
    plt.show()


num_epochs = 100 # Just for demo, adjust this higher.
training(myModel, train_dl, num_epochs)
torch.save(myModel, 'model.pt')
paint_training_acc(valid_acc,train_acc)
paint_training_loss(valid_loss,train_loss)
print(valid_acc)
print(valid_loss)
print(train_acc)
print(train_loss)
print("model is save!")
