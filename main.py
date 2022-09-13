import os
import shutil
import utils
import torch
import hubert
import librosa
import logging
import soundfile
import torchcrepe
import torchaudio
import numpy as np
from wav_temp import merge
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from pydub import AudioSegment

logging.getLogger('numba').setLevel(logging.WARNING)
hubert_soft = hubert.hubert_soft('pth/hubert.pt')


# python删除文件的方法 os.remove(path)path指的是文件的绝对路径,如：
def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data):  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)


def cut(cut_time, file_path, vocal_name, out_dir):
    audio_segment = AudioSegment.from_file(file_path, format='wav')

    total = int(audio_segment.duration_seconds / cut_time)  # 计算音频切片后的个数
    for i in range(total):
        # 将音频10s切片，并以顺序进行命名
        audio_segment[i * cut_time * 1000:(i + 1) * cut_time * 1000].export(f"{out_dir}/{vocal_name}-{i}.wav",
                                                                            format="wav")
    audio_segment[total * cut_time * 1000:].export(f"{out_dir}/{vocal_name}-{total}.wav", format="wav")  # 缺少结尾的音频片段


def resample_to_22050(audio_path):
    raw_audio, raw_sample_rate = torchaudio.load(audio_path)
    audio_22050 = torchaudio.transforms.Resample(orig_freq=raw_sample_rate, new_freq=22050)(raw_audio)[0]
    soundfile.write("./raw/vocals.wav", audio_22050, 22050)


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def resize2d(source, target_len):
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)),
                       source)
    return np.nan_to_num(target)


def convert_wav_22050_to_f0():
    if torch.cuda.is_available():
        audio, sr = torchcrepe.load.audio(source_path)
        tmp = torchcrepe.predict(audio=audio, fmin=50, fmax=550,
                                 sample_rate=22050, model='full',
                                 batch_size=2048, device='cuda:0').numpy()[0]
    else:
        tmp = librosa.pyin(librosa.load(source_path)[0],
                           fmin=librosa.note_to_hz('C2'),
                           fmax=librosa.note_to_hz('C7'),
                           frame_length=1780)[0]
    f0 = np.zeros_like(tmp)
    f0[tmp > 0] = tmp[tmp > 0]
    return f0


# 这个是config.json也换成自己的
hps_ms = utils.get_hparams_from_file("./configs/vctk_base.json")
net_g_ms = SynthesizerTrn(
    len(symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps_ms.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model)
_ = utils.load_checkpoint("pth/G.pth", net_g_ms, None)

# 每次合成长度，30s内，太高了爆掉显存(1066一次15s以内）
cut_time = 15
vc_transform = 1
speaker_id = 0

clean_name = "vocals"
bgm_name = "bgm"
out_audio_name = clean_name

resample_to_22050(f'./raw/{clean_name}.wav')
del_file("./wav_temp/input")
del_file("./wav_temp/output")

raw_audio_path = f"./raw/{clean_name}.wav"

audio, sample_rate = torchaudio.load(raw_audio_path)

audio_time = audio.shape[-1] / 22050
if audio_time > 1.3 * cut_time:
    cut(int(cut_time), raw_audio_path, "./wav_temp/input")
else:
    shutil.copy(f"./raw/{clean_name}.wav", f"./wav_temp/input/{clean_name}-0.wav")
file_list = os.listdir("./wav_temp/input")

count = 0
for file_name in file_list:
    source_path = "./wav_temp/input/" + file_name
    vc_transform = 1
    audio, sample_rate = torchaudio.load(source_path)
    input_size = audio.shape[-1]
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)[0]
    audio22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(audio)[0]

    # 此版本使用torchcrepe加速获取f0
    f0 = convert_wav_22050_to_f0()

    source = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        units = hubert_soft.units(source)
        soft = units.squeeze(0).numpy()
        f0 = resize2d(f0, len(soft[:, 0])) * int(vc_transform)
        soft[:, 0] = f0 / 10
    sid = torch.LongTensor([int(speaker_id)])
    stn_tst = torch.FloatTensor(soft)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0, noise_scale_w=0, length_scale=1)[0][
            0, 0].data.float().numpy()
    soundfile.write("./wav_temp/output/" + file_name, audio, int(audio.shape[0] / input_size * 22050))
    count += 1
    print("%s success: %.2f%%" % (file_name, 100 * count / len(file_list)))
merge.run(clean_name, bgm_name, out_audio_name)
