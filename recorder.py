import pyaudio
import wave
input_filename = 1               # 麦克风采集的语音输入
input_filepath = "record/"              # 输入文件的path
import time
import threading
p = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1                # 声道数
RATE = 22050                # 采样率
CHUNK = 256
RECORD_SECONDS = 2  # 录音时间
stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)


def save_audio(WAVE_OUTPUT_FILENAME,CHANNELS,FORMAT,RATE,CHUNK,RECORD_SECONDS):
        global audio_data
        last_audio_data=[]
        while True:
            if last_audio_data!=audio_data and audio_data!=[]:
                #print(len(audio_data))
                in_path = input_filepath + str(WAVE_OUTPUT_FILENAME)+".wav"
                wf = wave.open(in_path, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(audio_data))
                wf.close()
                WAVE_OUTPUT_FILENAME+=1
                last_audio_data=audio_data


def get_audio(stream,RECORD_SECONDS):
        global audio_data
        #a=time.time() 
        CHUNK = 256
        audio_data = []
        
        while True:
            frames=[]
            for i in range(0,int(RATE / CHUNK * RECORD_SECONDS)):
                #print(type(audio_data))
                data = stream.read(CHUNK)
                frames.append(data)
            audio_data=frames
            #print(len(audio_data))





threading.Thread(target=get_audio, args=(stream,RECORD_SECONDS,)).start()
print("准备开始录音")
time.sleep(RECORD_SECONDS)

threading.Thread(target=save_audio, args=(input_filename,CHANNELS,FORMAT,RATE,CHUNK,RECORD_SECONDS,)).start()
    

