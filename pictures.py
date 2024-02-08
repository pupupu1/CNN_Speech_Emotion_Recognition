# - * - coding: utf-8 - * -
import librosa
import librosa.display
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

plt.figure(dpi=600) # 将显示的所有图分辨率调高
matplotlib.rc("font",family='SimHei') # 显示中文
matplotlib.rcParams['axes.unicode_minus']=False # 显示符号

path = 'wav/03b02La.wav'

def displayWaveform(): # 显示语音时域波形
    """
    display waveform of a given speech sample
    :param sample_name: speech sample name
    :param fs: sample frequency
    :return:
    """
    samples, sr = librosa.load(path, sr=16000)
    # samples = samples[6000:16000]

    print(len(samples), sr)
    time = np.arange(0, len(samples)) * (1.0 / sr)

    plt.plot(time, samples)
    plt.title("“高兴”情感的语音信号波形图")
    plt.xlabel("时长（秒）")
    plt.ylabel("振幅")
    plt.savefig("语音信号波形图-高兴", dpi=600)
    plt.show()

def displaySpectrum(): # 显示语音频域谱线
    x, sr = librosa.load(path, sr=16000)
    print(len(x))
    # ft = librosa.stft(x)
    # magnitude = np.abs(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
    # frequency = np.angle(ft)  # (0, 16000, 121632)

    ft = fft(x)
    print(len(ft), type(ft), np.max(ft), np.min(ft))
    magnitude = np.absolute(ft)  # 对fft的结果直接取模（取绝对值），得到幅度magnitude
    frequency = np.linspace(0, sr, len(magnitude))  # (0, 16000, 121632)

    print(len(magnitude), type(magnitude), np.max(magnitude), np.min(magnitude))
    print(len(frequency), type(frequency), np.max(frequency), np.min(frequency))

    # plot spectrum，限定[:40000]
    #plt.figure(figsize=(18, 8))
    plt.plot(frequency[:10000], magnitude[:10000])  # magnitude spectrum
    plt.title("“高兴”情感的语音信号频谱图")
    plt.xlabel("频率（赫兹）")
    plt.ylabel("幅度")
    plt.savefig("语音信号频谱图-高兴", dpi=600)
    plt.show()

    # # plot spectrum，不限定 [对称]
    # plt.figure(figsize=(18, 8))
    # plt.plot(frequency, magnitude)  # magnitude spectrum
    # plt.title("语音信号频域谱线")
    # plt.xlabel("频率（赫兹）")
    # plt.ylabel("幅度")
    # plt.show()


def displaySpectrogram():
    x, sr = librosa.load(path, sr=16000)

    # compute power spectrogram with stft(short-time fourier transform):
    # 基于stft，计算power spectrogram
    spectrogram = librosa.amplitude_to_db(librosa.stft(x))

    # show
    librosa.display.specshow(spectrogram, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('“高兴”情感的语音信号对数谱图')
    plt.xlabel('时长（秒）')
    plt.ylabel('频率（赫兹）')
    plt.savefig("语音信号对数谱图-高兴", dpi=600)
    plt.show()


if __name__ == '__main__':
    displayWaveform()
    displaySpectrum()
    displaySpectrogram()