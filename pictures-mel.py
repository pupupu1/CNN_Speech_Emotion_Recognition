import librosa.display
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.figure(dpi=600) # 将显示的所有图分辨率调高
matplotlib.rc("font",family='SimHei') # 显示中文
matplotlib.rcParams['axes.unicode_minus']=False # 显示符号

path = 'wav/03a02Fc.wav'

# sr=None声音保持原采样频率， mono=False声音保持原通道数
data, fs = librosa.load(path, sr=None, mono=False)

L = len(data)
print('Time:', L / fs)

# 归一化
data = data * 1.0 / max(data)

# 0.025s
framelength = 0.025
# NFFT点数=0.025*fs
framesize = int(framelength * fs)
print("NFFT:", framesize)

#提取mel特征
mel_spect = librosa.feature.melspectrogram(data, sr=fs, n_fft=framesize)
#转化为log形式
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

#画mel谱图
librosa.display.specshow(mel_spect, sr=fs, x_axis='time', y_axis='mel')
plt.ylabel('梅尔频率')
plt.xlabel('时间(秒)')
plt.title('“高兴”情感的梅尔谱图')
plt.show()
