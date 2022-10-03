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


def save_audio(WAVE_OUTPUT_FILENAME,CHANNELS,FORMAT,RATE):
        global audio_data
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_data))
        wf.close()
        
def get_audio(stream,RECORD_SECONDS):
        global audio_data
        #a=time.time() 
        CHUNK = 256



        #print("*"*10, "开始录音：请在5秒内输入语音")
        frames = []
        frames_count=0
        lenth=int(RATE / CHUNK * RECORD_SECONDS)
        while TRUE:
            data = stream.read(CHUNK)
            frames.append(data)
            frames_count+=1
            if frames_count == int(RATE / CHUNK * RECORD_SECONDS):
              audio_data=frames
              frames=[]



        
        #stream.stop_stream()
        #stream.close()
        #p.terminate()
        #print(time.time()-a)

        #print(time.time()-a)



threading.Thread(target=get_audio, args=(stream,RECORD_SECONDS,)).start()
print("准备开始录音")
time.sleep(RECORD_SECONDS)
last_audio_data=[]
while True:
    in_path = input_filepath + str(input_filename)+".wav"
    if last_audio_data!=audio_data:
        save_audio(in_path,CHANNELS,FORMAT,RATE)
        input_filename+=1
        last_audio_data=audio_data
