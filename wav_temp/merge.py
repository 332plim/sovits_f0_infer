import os
from pydub import AudioSegment


def add_db(music1, music2):
    music1_db = music1.dBFS
    music2_db = music2.dBFS
    # 调整两个音频的响度一致
    dbplus = music1_db - music2_db
    if dbplus > 0:
        music2 += abs(dbplus)
    elif dbplus < 0:
        music2 -= abs(dbplus)
    return music2


def wav_mix(file_name):
    bgm = AudioSegment.from_wav("./raw/bgm.wav")
    vits = AudioSegment.from_wav("./results/out_vits.wav")
    vocals = AudioSegment.from_wav("./raw/vocals.wav")
    vits = add_db(vocals, vits)
    # mix sound2 with sound1, starting at 5000ms into sound1)
    output = bgm.overlay(vits)
    # save the result
    output.export("./results/mix_%s.mp3" % file_name, format="mp3")


def wav_combine(*args):
    n = args[0][0]  # 需要拼接的wav个数
    i = 1
    sounds = []
    while i <= n:
        sounds.append(AudioSegment.from_wav(args[0][i]))
        i += 1
    playlist = AudioSegment.empty()
    for sound in sounds:
        playlist += sound
    playlist.export(args[0][n + 1], format="wav")


def run(file_name):
    file_list = os.listdir("./wav_temp/output")
    in_files = [len(file_list)]
    for i in range(0, len(file_list)):
        in_files.append("./wav_temp/output/vocals-%s.wav" % i)
    out_path = './results/out_vits.wav'
    in_files.append(out_path)
    wav_combine(in_files)
    print("out vits success")
    if os.path.exists("./raw/bgm.wav"):
        wav_mix(file_name)
        print("out song success")


if __name__ == "__main__":
    run("demo.wav")
