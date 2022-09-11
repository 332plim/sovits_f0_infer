import os
import utils
import torch
import hubert
import logging
import soundfile
import torchcrepe
import torchaudio
import numpy as np
from wav_temp import merge, cut
from models import SynthesizerTrn

logging.getLogger('numba').setLevel(logging.WARNING)
hubert_soft = hubert.hubert_soft('pth/hubert.pt')

# 这个是config.json也换成自己的
hps_ms = utils.get_hparams_from_file("./configs/vctk_base.json")
net_g_ms = SynthesizerTrn(
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps_ms.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model)
_ = utils.load_checkpoint("pth/G.pth", net_g_ms, None)


def resize2d(source, target_len):
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source), len(source) / target_len), np.arange(0, len(source)), source)
    return np.nan_to_num(target)


def convert_wav_22050_to_f0():
    audio, sr = torchcrepe.load.audio(source_path)
    tmp = torchcrepe.predict(audio=audio, fmin=50, fmax=550,
                             sample_rate=22050, model='full',
                             batch_size=2048, device='cuda:0').numpy()[0]
    f0 = np.zeros_like(tmp)
    f0[tmp > 0] = tmp[tmp > 0]
    return f0


# 每次合成长度，30s内，太高了爆掉显存
cut_time = 8
file_name = "vocals.wav"
cut.cut_audios(cut_time)
file_list = os.listdir("./wav_temp/input")
count = 0
for file_name in file_list:
    source_path = "./wav_temp/input/" + file_name
    vc_transform = 1
    audio, sample_rate = torchaudio.load(source_path)

    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)[0]
    audio22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(audio)[0]

    # 此版本使用torchcrepe加速获取f0
    f0 = convert_wav_22050_to_f0()

    source = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        units = hubert_soft.units(source)
        soft = units.squeeze(0).numpy()
        f0 = resize2d(f0, len(soft[:, 0])) * vc_transform
        soft[:, 0] = f0 / 10
    sid = torch.LongTensor([0])
    stn_tst = torch.FloatTensor(soft)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0, noise_scale_w=0, length_scale=1)[0][
            0, 0].data.float().numpy()
    soundfile.write("./wav_temp/output/" + file_name, audio, int(audio.shape[0] / cut_time))
    count += 1
    print("%s success: %.2f%%" % (file_name, 100 * count / len(file_list)))
merge.run("vocals")
