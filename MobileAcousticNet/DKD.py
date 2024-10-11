import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from plotTest.AcousticNet_reduced import device,myModel
from GetData2 import train_dl,val_dl

def get_labels_mask(output,labels):
    labels = labels.reshape(-1)
    mask = torch.zeros_like(output).scatter_(1, labels.unsqueeze(1),1).bool()
    return mask


def get_other_mask(output, labels):
    labels = labels.reshape(-1)
    mask = torch.ones_like(output).scatter_(1, labels.unsqueeze(1), 0).bool()
    return mask


def cat_mask(pred, mask1, mask2):
    pred1 = (pred * mask1).sum(dim=1,keepdims=True)
    pred2 = (pred * mask2).sum(dim=1,keepdims=True)
    pred = torch.cat([pred1,pred2],dim = 1)
    return pred


train_loss = []
valid_loss = []
train_acc = []
valid_acc = []

def DKD_Loss(student_output,teacher_output,labels,alpha,beta,temperature):
    # print(student_output)
    # print(teacher_output)
    labels_mask = get_labels_mask(student_output,labels)
    # print(labels_mask)
    other_mask = get_other_mask(student_output,labels)
    # print(other_mask) # 为什么开始的时候other_mask的值全部是0，把ones_like错误的写成了ones_like

    pred_student = F.softmax(student_output/temperature,dim=1)
    # print(pred_student)
    pred_teacher = F.softmax(teacher_output/temperature,dim=1)
    # print(pred_teacher)
    pred_student = cat_mask(pred_student,labels_mask,other_mask)
    # print(pred_student)
    pred_teacher = cat_mask(pred_teacher,labels_mask,other_mask)
    # print(pred_teacher)

    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student,pred_teacher,size_average=False)
        *(temperature**2)
        /labels.shape[0]
    )

    pred_teacher_part2 = F.softmax(
        teacher_output/temperature - 1000.0 * labels_mask, dim = 1
    )
    log_pred_student_part2 = F.log_softmax(
        student_output/temperature - 1000.0 * labels_mask, dim = 1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2,pred_teacher_part2,size_average = False)
        *(temperature**2)
        /labels.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss

def DKD_training(teacher,student,train_dl,val_dl,num_epochs):
    T = 7
    CE_WEIGHT = 1.0
    ALPHA = 0.8
    BETA = 0.2
    WARMUP = 20

    hard_loss = nn.BCELoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer= optimizer, max_lr=0.001,
        steps_per_epoch = int(len(train_dl)),
        epochs = num_epochs, anneal_strategy="linear"
    )

    for epoch in range(num_epochs):

        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        val_loss = 0.0

        student.train()

        for i, data in enumerate(train_dl):

            inputs, labels = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m)/inputs_s

            with torch.no_grad():
                teacher_output = teacher(inputs)
                teacher_output = F.sigmoid(teacher_output)
                # teacher_output_loss = teacher_output[:, 1]

            optimizer.zero_grad()
            student_output = student(inputs)
            student_output = F.sigmoid(student_output)
            student_output_loss = student_output[:, 1]
            loss_stu = CE_WEIGHT * hard_loss(student_output_loss, labels.float())
            loss_dkd = DKD_Loss(
                student_output, teacher_output,
                labels, ALPHA, BETA, T
            )
            loss_bsc = student.reg_loss(alpha=0.03)

            loss = loss_stu + loss_bsc + 0.01*loss_dkd

            # print(loss_stu)
            # print(loss_bsc)
            # print(loss_dkd)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, prediction = torch.max(student_output,1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        num_batches = len(train_dl)
        loss = running_loss / num_batches
        acc = correct_prediction / total_prediction

        train_loss.append(loss)
        train_acc.append((acc))
        print(f'Epoch: {epoch}, Loss: {loss:.6f}, Accuracy: {acc:.2f}')


        for i, data in enumerate(val_dl):

            inputs, labels = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m)/inputs_s

            with torch.no_grad():
                teacher_output = teacher(inputs)
                teacher_output = F.sigmoid(teacher_output)

            optimizer.zero_grad()
            student_output = student(inputs)
            student_output = F.sigmoid(student_output)
            student_output_loss = student_output[:, 1]
            loss_stu = CE_WEIGHT * hard_loss(student_output_loss, labels.float())
            loss_dkd = DKD_Loss(
                student_output, teacher_output,
                labels, ALPHA, BETA, T
            )
            loss_bsc = student.reg_loss(alpha=0.03)

            loss = loss_stu + loss_bsc + 0.01 * loss_dkd

            # print(loss_stu)
            # print(loss_bsc)
            # print(loss_dkd)

            val_loss += loss.item()
            _, prediction = torch.max(student_output,1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]


        num_batches = len(val_dl)
        avg_loss = val_loss / num_batches
        acc = correct_prediction / total_prediction

        valid_acc.append(acc)
        valid_loss.append(avg_loss)
        print(f'Epoch: {epoch}, Loss_val: {avg_loss:}, Accuracy: {acc:}')

import numpy as np
def paint_training_acc(val,train):
    x = range(0,100)
    plt.figure(figsize=(15,4))
    plt.xticks(np.arange(0,100,5))
    plt.plot(x,val,label="valSet",linewidth=1.5,color='b')
    plt.plot(x,train,label="trainSet",linewidth=1.5,color='g')
    plt.xlabel("epoch")
    # plt.ylabel("val/train")
    plt.legend(loc="upper right")
    plt.savefig("crickets_acc")
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
    plt.savefig("crickets_loss")
    plt.show()

# 加载已经训练好的Teacher
Teacher = torch.load("Models/net_modify/insects_teacher.pt")
#
num_epochs = 100
#
DKD_training(teacher=Teacher, student=myModel ,train_dl=train_dl,val_dl=val_dl,num_epochs=num_epochs)
torch.save(myModel, 'Models/dkd/insects.pt')
paint_training_acc(valid_acc,train_acc)
paint_training_loss(valid_loss,train_loss)
print(valid_acc)
print(valid_loss)
print(train_acc)
print(train_loss)
print("model is save!")