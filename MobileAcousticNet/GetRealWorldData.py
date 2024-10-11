import pandas as pd
import glob
import torch
from torch.utils.data import random_split
from Dataloader import SoundDS

df_dataset = pd.DataFrame(columns=['path', 'ID'])

filelist_clean = list(glob.glob('D:/AudioDetectionOfXylophagous/clean/*.wav'))
filelist_infested = list(glob.glob('D:/AudioDetectionOfXylophagous/infested/*.wav'))

dict = {'path': filelist_clean, 'ID': 0}
df_dataset_part1 = pd.DataFrame(dict)
dict = {'path': filelist_infested, 'ID': 1}
df_dataset_part2 = pd.DataFrame(dict)

df_dataset = df_dataset._append(df_dataset_part1)
df_dataset = df_dataset._append(df_dataset_part2,ignore_index=True)

df_dataset = df_dataset.sample(frac=1).reset_index(drop=True)
Datas = SoundDS(df_dataset,'')
train_datas, val_datas, test_datas = random_split(dataset=Datas, lengths=[0.6,0.2,0.2],generator=torch.Generator().manual_seed(0))

train_dl = torch.utils.data.DataLoader(train_datas, batch_size=16, shuffle=True,drop_last=True)
val_dl = torch.utils.data.DataLoader(val_datas, batch_size=16, shuffle=True,drop_last=True)
test_dl = torch.utils.data.DataLoader(test_datas, batch_size=16, shuffle=True,drop_last=True)

