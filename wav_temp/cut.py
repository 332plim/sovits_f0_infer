import os
import wave
import numpy as np
import pylab as plt
import librosa
import torch
import torchaudio
import soundfile
import shutil


# python删除文件的方法 os.remove(path)path指的是文件的绝对路径,如：
def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data):  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)


def resample():
    source_path = './raw/vocals.wav'
    audio, sample_rate = torchaudio.load(source_path)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    audio22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(audio)[0]
    soundfile.write("./raw/vocals.wav", audio22050, 22050)


def cut_audios(cut_time):
    del_file("./wav_temp/input")
    del_file("./wav_temp/output")
    resample()
    path = './raw/'
    file_name = "vocals.wav"
    f = wave.open(path + file_name, 'rb')
    params = f.getparams()  # 读取音频文件信息
    nchannels, sampwidth, framerate, nframes = params[:4]  # 声道数, 量化位数, 采样频率, 采样点数
    str_data = f.readframes(nframes)
    f.close()

    wave_data = np.frombuffer(str_data, dtype=np.short)
    # 根据声道数对音频进行转换
    if nchannels > 1:
        wave_data.shape = -1, 2
        wave_data = wave_data.T
        temp_data = wave_data.T
    else:
        wave_data = wave_data.T
        temp_data = wave_data.T

    cut_frame_num = framerate * cut_time
    cut_num = nframes / cut_frame_num  # 音频片段数
    step_num = int(cut_frame_num)

    for j in range(int(cut_num)):
        file_name = "./wav_temp/input/vocals-%s.wav" % j
        temp_data_temp = temp_data[step_num * j:step_num * (j + 1)]
        temp_data_temp.shape = 1, -1
        temp_data_temp = temp_data_temp.astype(np.short)  # 打开WAV文档
        f = wave.open(file_name, 'wb')
        # 配置声道数、量化位数和取样频率
        f.setnchannels(nchannels)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        f.writeframes(temp_data_temp.tostring())  # 将wav_data转换为二进制数据写入文件
        f.close()
