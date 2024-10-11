from Transform import AudioUtil
from torch.utils.data import Dataset


class SoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 10000
        self.sr = 8000
        self.channel = 1
        self.shift_pct = 0.4

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        audio_file = self.data_path + str(self.df.loc[idx, 'path'])
        #print(audio_file)

        class_id = self.df.loc[idx, 'ID']


        aud = AudioUtil.open(audio_file)

        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)

        input_feature = AudioUtil.spectro_gram(dur_aud, n_mels=13, n_fft=1024, hop_len=None)
        # aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        return input_feature, class_id
# 在添加背景噪声后，得到的新数据，是否可以用于进一步的鲁棒性检测？