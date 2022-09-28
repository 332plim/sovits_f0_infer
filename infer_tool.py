import os
import torch
import hubert
import soundfile
import torchaudio
import numpy as np
from pydub import AudioSegment


def resize2d_f0(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)),
                       source)
    res = np.nan_to_num(target)
    return res


# python删除文件的方法 os.remove(path)path指的是文件的绝对路径,如：
def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        os.remove(path_data + i)


def cut(c_time, file_path, vocal_name, out_dir):
    audio_segment = AudioSegment.from_file(file_path, format='wav')

    total = int(audio_segment.duration_seconds / c_time)  # 计算音频切片后的个数
    for i in range(total):
        # 将音频10s切片，并以顺序进行命名
        audio_segment[i * c_time * 1000:(i + 1) * c_time * 1000].export(f"{out_dir}/{vocal_name}-{i}.wav",
                                                                        format="wav")
    audio_segment[total * c_time * 1000:].export(f"{out_dir}/{vocal_name}-{total}.wav", format="wav")  # 缺少结尾的音频片段


def resample_to_22050(audio_path):
    raw_audio, raw_sample_rate = torchaudio.load(audio_path)
    audio_22050 = torchaudio.transforms.Resample(orig_freq=raw_sample_rate, new_freq=22050)(raw_audio)[0]
    soundfile.write(audio_path, audio_22050, 22050)


def resize2d_plus(source, target_len):
    source = source.astype(float)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)),
                       source)
    res = np.nan_to_num(target)
    ret = res[:].astype(int)
    # 若调整大小时采样到中间的点，则以上一个点作为当前音高值
    for i in range(len(res)):
        if res[i] - ret[i] > 0.001:
            ret[i] = ret[i - 1]
    return ret


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for i in range(0, len(b) - len(a)):
            a.append(a[0])
