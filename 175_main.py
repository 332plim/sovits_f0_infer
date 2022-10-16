#adapted to onnx model from https://huggingface.co/spaces/14-26AA/sovits_aishell3
import onnxruntime
import numpy as np
import pyworld as pw
import librosa
import soundfile as sf
import sounddevice as sd


import time
import threading
import os
import infer_tool

#说话人ID（1-175）
sid=22


#移调
vc_transform=-4

name=1


infer_tool.del_file("./record/")
threading.Thread(target=os.system, args=("python sclient_split_recorder.py",)).start()


def play(data,fs):
    sd.play(data, fs)
    #print("played")

    status = sd.wait()

def resize2d(source, target_len):
    source[source<0.001] = np.nan
    target = np.interp(np.linspace(0, len(source)-1, num=target_len,endpoint=True), np.arange(0, len(source)), source)
    return np.nan_to_num(target)

def _calculate_f0(input: np.ndarray,length,sr,f0min,f0max,
                      use_continuous_f0: bool=True,
                      use_log_f0: bool=True) -> np.ndarray:
        input = input.astype(float)
        frame_period = len(input)/sr/(length)*1000
        f0, timeaxis = pw.dio(
            input,
            fs=sr,
            f0_floor=f0min,
            f0_ceil=f0max,
            frame_period=frame_period)
        f0 = pw.stonemask(input, f0, timeaxis, sr)
        if use_log_f0:
            nonzero_idxs = np.where(f0 != 0)[0]
            f0[nonzero_idxs] = np.log(f0[nonzero_idxs])
        return f0.reshape(-1)


def get_text(wav,sr,transform=1.0):
    global hubertsession
    #wav, sr = librosa.load(file,sr=None)
    if len(wav.shape) > 1:
        wav = librosa.to_mono(wav.transpose(1, 0)) 
    if sr!=16000:  
        wav16 = librosa.resample(wav, sr, 16000)
    else:
        wav16=wav
    
    source = {"source":np.expand_dims(np.expand_dims(wav16,0),0)}
    
    units = np.array(hubertsession.run(['embed'], source)[0])
    f0=_calculate_f0(wav,units.shape[1],sr,
            f0min=librosa.note_to_hz('C2'),
            f0max=librosa.note_to_hz('C7'))
    f0=resize2d(f0,units.shape[1])
    f0[f0!=0]=f0[f0!=0]+np.log(transform)
    expf0 = np.expand_dims(f0,(0,2))
    output=np.concatenate((units,expf0,expf0),axis=2)
    return output.astype(np.float32),f0

def getkey(key):
    return np.power(2,key/12.0)

def infer(f,reqf0=False):
    global sid,infersession,x,sourcef0,x_lengths
    #speaker=int(speaker[7:])

    file=f
    audio,sr = librosa.load(file,sr=None)
    x,sourcef0 = get_text(audio,sr,getkey(vc_transform))
    x_lengths = [np.size(x,1)]
    duration = audio.shape[0] / sr

    #print(audio,sr,duration)
    #if duration > 120:
    #    return "请上传小于2min的音频", None
    #audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)

    #print(x_lengths[0],sr,sid,key)
    #sid = [speaker]

    ort_inputs = {'x':x,'x_lengths':x_lengths,'sid':[sid],"noise_scale":[0.667],"length_scale":[1.0],"noise_scale_w":[0.8]} 
    ort_output = infersession.run(['audio'], ort_inputs)
    #sf.write("audio.wav",ort_output[0][0][0],22050,'PCM_16',format='wav')

    genf0=np.array([])
    if reqf0:
        wav, sr = librosa.load(o,sr=None)
        genf0=_calculate_f0(wav,x_lengths[0],sr,
            f0min=librosa.note_to_hz('C2'),
            f0max=librosa.note_to_hz('C7'))
        genf0=resize2d(genf0,x_lengths[0])
    #print( 'success',(22050,ort_output[0][0][0]))#sourcef0.tolist(),genf0.tolist()
    return ort_output

infersession = onnxruntime.InferenceSession("pth/onnxmodel334.onnx",providers=['CUDAExecutionProvider'])
hubertsession = onnxruntime.InferenceSession("pth/hubert.onnx",providers=['CUDAExecutionProvider'])

while True:
    a=time.time()
    if os.path.exists("record/"+str(name)+".wav"):
        clean_names=[str(name)]
    else:
        continue
    try:
        ort_output=infer("record/"+str(name)+".wav")
        print("time taken: "+str(time.time()-a))
        print("playing",name)
        play(ort_output[0][0][0],22050)
        print("play end",name)
    except Exception as e:
        print(e)
    os.system("del record\\"+str(name)+".wav")
    name+=1

        
