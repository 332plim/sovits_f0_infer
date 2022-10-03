import logging
import os
import shutil

import demjson
import soundfile
import torch
import torchaudio

import hubert_model
import infer_tool
import utils
from models import SynthesizerTrn
from preprocess_wave import FeatureInput
from wav_temp import merge

logging.getLogger('numba').setLevel(logging.WARNING)


def get_units(path):
    source, sr = torchaudio.load(path)
    source = torchaudio.functional.resample(source, sr, 16000)
    source = source.unsqueeze(0).to(dev)
    with torch.inference_mode():
        units = hubert_soft.units(source)
        return units


def transcribe(path, length, transform):
    feature_pit = featureInput.compute_f0(path)
    feature_pit = feature_pit * 2 ** (transform / 12)
    feature_pit = infer_tool.resize2d_f0(feature_pit, length)
    coarse_pit = featureInput.coarse_f0(feature_pit)
    return coarse_pit


# 自行创建pth文件夹，放置hubert、sovits模型，创建raw、results文件夹
# 可填写音源文件列表，音源文件格式为单声道22050采样率wav，放置于raw文件夹下
clean_names = ["多声线测试"]
# bgm、trans分别对应歌曲列表，若能找到相应文件、则自动合并伴奏，若找不到bgm，则输出干声（不使用bgm合成多首歌时，可只随意填写一个不存在的bgm名）
bgm_names = ["bgm1"]
# 合成多少歌曲时，若半音数量不足、自动补齐相同数量（按第一首歌的半音）
trans = [-3]  # 加减半音数（可为正负）s
# 每首歌同时输出的speaker_id
id_list = [2]

# 每次合成长度，建议30s内，太高了爆掉显存(gtx1066一次15s以内）
cut_time = 60
model_name = "530_epochs"  # 模型名称（pth文件夹下）
config_name = "yilanqiu.json"  # 模型配置（config文件夹下）

# 自行下载hubert-soft-0d54a1f4.pt改名为hubert.pt放置于pth文件夹下
# https://github.com/bshall/hubert/releases/tag/v0.1
hubert_soft = hubert_model.hubert_soft('pth/hubert.pt')

# 以下内容无需修改
hps_ms = utils.get_hparams_from_file(f"configs/{config_name}")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_g_ms = SynthesizerTrn(
    178,
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps_ms.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model)
_ = utils.load_checkpoint(f"pth/{model_name}.pth", net_g_ms, None)
_ = net_g_ms.eval().to(dev)

featureInput = FeatureInput(hps_ms.data.sampling_rate, hps_ms.data.hop_length)
# 自动补齐
infer_tool.fill_a_to_b(bgm_names, clean_names)
infer_tool.fill_a_to_b(trans, clean_names)
for clean_name, bgm_name, tran in zip(clean_names, bgm_names, trans):
    infer_tool.resample_to_22050(f'./raw/{clean_name}.wav')
    for speaker_id in id_list:
        speakers = demjson.decode_file(f"configs/{config_name}")["speakers"]
        out_audio_name = model_name + f"_{clean_name}_{speakers[speaker_id]}"
        # 清除缓存文件
        infer_tool.del_file("./wav_temp/input/")
        infer_tool.del_file("./wav_temp/output/")

        raw_audio_path = f"./raw/{clean_name}.wav"
        audio, sample_rate = torchaudio.load(raw_audio_path)

        audio_time = audio.shape[-1] / 22050
        if audio_time > 1.3 * int(cut_time):
            infer_tool.cut(int(cut_time), raw_audio_path, out_audio_name, "./wav_temp/input")
        else:
            shutil.copy(f"./raw/{clean_name}.wav", f"./wav_temp/input/{out_audio_name}-0.wav")
        file_list = os.listdir("./wav_temp/input")

        count = 0
        for file_name in file_list:
            source_path = "./wav_temp/input/" + file_name
            audio, sample_rate = torchaudio.load(source_path)
            input_size = audio.shape[-1]

            sid = torch.LongTensor([int(speaker_id)]).to(dev)
            soft = get_units(source_path).squeeze(0).cpu().numpy()
            pitch = transcribe(source_path, soft.shape[0], tran)
            pitch = torch.LongTensor(pitch).unsqueeze(0).to(dev)
            stn_tst = torch.FloatTensor(soft)
            with torch.no_grad():
                x_tst = stn_tst.unsqueeze(0).to(dev)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(dev)
                audio = \
                    net_g_ms.infer(x_tst, x_tst_lengths, pitch, sid=sid, noise_scale=.3, noise_scale_w=0.5,
                                   length_scale=1)[0][
                        0, 0].data.float().cpu().numpy()

            soundfile.write("./wav_temp/output/" + file_name, audio, int(audio.shape[0] / input_size * 22050))
            count += 1
            print("%s success: %.2f%%" % (file_name, 100 * count / len(file_list)))
        merge.run(out_audio_name, bgm_name, out_audio_name)
