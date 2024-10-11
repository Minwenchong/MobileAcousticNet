import pandas as pd
import glob
import torch
from torch.utils.data import random_split
from Dataloader import SoundDS

df_train = pd.DataFrame(columns=['path', 'ID'])
filelist1 = list(glob.glob('Datasets\\realdata\\train\\clean\\*.wav'))
filelist2 = list(glob.glob('Datasets\\realdata\\train\\infested\\*.wav'))
dict = {'path': filelist1, 'ID': 0}
df_train_part1 = pd.DataFrame(dict)
dict = {'path': filelist2, 'ID': 1}
df_train_part2 = pd.DataFrame(dict)
df_train = df_train._append(df_train_part1)
df_train = df_train._append(df_train_part2, ignore_index=True)
df_train = df_train.sample(frac=1).reset_index(drop=True)
TrainDatas = SoundDS(df_train, '')
train_dl = torch.utils.data.DataLoader(TrainDatas, batch_size=16, shuffle=True,drop_last=True)


df_vaild = pd.DataFrame(columns=['path', 'ID'])
filelist3 = list(glob.glob('Datasets\\realdata\\valid\\clean\\*.wav'))
filelist4 = list(glob.glob('Datasets\\realdata\\valid\\infested\\*.wav'))
dict = {'path': filelist3, 'ID': 0}
df_test_part1 = pd.DataFrame(dict)
dict = {'path': filelist4, 'ID': 1}
df_test_part2 = pd.DataFrame(dict)
df_vaild = df_vaild._append(df_test_part1)
df_vaild = df_vaild._append(df_test_part2, ignore_index=True)
ValDatas = SoundDS(df_vaild, '')
val_dl = torch.utils.data.DataLoader(ValDatas, batch_size=16, shuffle=True,drop_last=True)


df_test = pd.DataFrame(columns=['path', 'ID'])
filelist3 = list(glob.glob('Datasets\\realdata\\test\\clean\\*.wav'))
filelist4 = list(glob.glob('Datasets\\realdata\\test\\infested\\*.wav'))
dict = {'path': filelist3, 'ID': 0}
df_test_part1 = pd.DataFrame(dict)
dict = {'path': filelist4, 'ID': 1}
df_test_part2 = pd.DataFrame(dict)
df_test = df_test._append(df_test_part1)
df_test = df_test._append(df_test_part2, ignore_index=True)
TestDatas = SoundDS(df_test, '')
test_dl = torch.utils.data.DataLoader(TestDatas, batch_size=16, shuffle=True,drop_last=True)

