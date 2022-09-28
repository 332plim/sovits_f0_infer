import os
import numpy as np
import icassp2022_vocal_transcription


def resize2d(source, target_len):
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


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))

    return file_lists


folder = "val"
wav_paths = get_end_file(f"./qiu/wavs/{folder}/", "wav")
for wav_path in wav_paths:
    pitch = icassp2022_vocal_transcription.transcribe(wav_path)
    soft = np.load(wav_path.replace("wavs", "soft").replace(".wav", ".npy"))
    pitch = resize2d(pitch, len(soft[:, 0]))
    np.save(wav_path.replace("wavs", "pitch").replace(".wav", ".npy"), pitch)
