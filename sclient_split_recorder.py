#-*- coding: utf-8 -*-
import os
import wave
from time import sleep
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
 
SUCCESS = 0
FAIL = 1
 
audio = pyaudio.PyAudio()
audio2 = ""
stream2 = ""
FORMAT = pyaudio.paInt16
stream = audio.open(format=FORMAT,
                    channels=1,
                    rate=22050,
                    input=True,
                    frames_per_buffer=256
                    )
 
 
"""
语音和噪声的区别可以体现在他们的能量上，语音段的能量比噪声段的能量大，如果环境噪声和系统输入的噪声比较小，
只要计算输入信号的短时能量就能够把语音段和噪声背景区分开，除此之外，用基于能量的算法来检测浊音通常效果也是比较理想的，
因为浊音的能量值比清音大得多，可以判断浊音和清音之间过渡的时刻[3]，但对清音来说，效果不是很好，因此还需要借助短时过零率来表征。 
短时能量可以近似为互补的情况，短时能量大的地方过零率小，短时能量小的地方过零率较大。  
基于短时能量和过零率的检测方法  尽管基于短时能量和过零率的检测方法各有其优缺点，但是若将这两种基本方法相结合起来使用也可以实现对语音信号可靠的端点检测。
无声段的短时能量为零，清音段的短时能量又比浊音段的短时能量大，
而在过零率方面，理想的情况是无声段的过零率为零，浊音段的过零率比清音段的过零率要大的多，
假设有一段语音，
如果某部分短时能量和过零率都为零或者为很小的值，就可以认为这部分为无声段，
如果该部分语音短时能量很大但是过零率很小，则认为该部分语音为浊音段，
如果该部分短时能量很小但是过零率很大，则认为该部分语音为清音段。
正如前面提到，语音信号具有短时性，因此在对语音信号进行分析时，
，需要将语音信号以30ms为一段分为若干帧来进行分析，则两帧起始点之间的间隔为10ms。
短时能量，无声<浊音<清音
过零率，无声<清音<浊音
"""
 
"""
事迹测试
说话时的过零率 在0-3之间   轻呼吸时   重呼吸呼吸时的过零率在2-7之间
说话时的短时能量 在940-12000之间    轻呼吸时   重呼吸15000-23000之间
"""
 
 
# 需要添加录音互斥功能能,某些功能开启的时候录音暂时关闭
def ZCR(curFrame):
    # 过零率
    tmp1 = curFrame[:-1]
    tmp2 = curFrame[1:]
    sings = (tmp1 * tmp2 <= 0)
    diffs = (tmp1 - tmp2) > 0.02
    zcr = np.sum(sings * diffs)
    return zcr
 
 
def STE(curFrame):
    # 短时能量
    amp = np.sum(np.abs(curFrame))
    return amp
 
 
class Vad(object):
    def __init__(self):
        # 初始短时能量高门限
        self.amp1 = 940
        # 初始短时能量低门限
        self.amp2 = 120
        # 初始短时过零率高门限
        self.zcr1 = 30
        # 初始短时过零率低门限
        self.zcr2 = 2
        # 允许最大静音长度
        self.maxsilence = 200     #允许换气的最长时间
        # 语音的最短长度
        self.minlen = 40        #  过滤小音量
        # 偏移值
        self.offsets = 40
        self.offsete = 40
        # 能量最大值
        self.max_en = 20000
        # 初始状态为静音
        self.status = 0
        self.count = 0
        self.silence = 0
        self.frame_len = 256
        self.frame_inc = 128
        self.cur_status = 0
        self.frames = []
        # 数据开始偏移
        self.frames_start = []
        self.frames_start_num = 0
        # 数据结束偏移
        self.frames_end = []
        self.frames_end_num = 0
        # 缓存数据
        self.cache_frames = []
        self.cache = ""
        # 最大缓存长度
        self.cache_frames_num = 0
        self.end_flag = False
        self.wait_flag = False
        self.on = True
        self.callback = None
        self.callback_res = []
        self.callback_kwargs = {}
 
        self.frames = []
        self.x = []
        self.y = []
 
    def check_ontime(self,cache_frame):  # self.cache的值为空   self.cache_frames的数据长度为744
 
        global audio2,stream2,name_num,speaker
        wave_data = np.frombuffer(cache_frame, dtype=np.int16)  # 这里的值竟然是256
        wave_data = wave_data * 1.0 / self.max_en  # max_en  为20000
        data = wave_data[np.arange(0, self.frame_len)]  # 取前frame_len个值   这个值为256
        # speech_data = self.cache_frames.pop(0)    #删除第一个元素，并把第一个元素给speech_data  ,长度为256
        # 获得音频过零率
        zcr = ZCR(data)
        # 获得音频的短时能量, 平方放大
        amp = STE(data)**2
        # 返回当前音频数据状态
        res = self.speech_status(amp, zcr)
 
        self.cur_status = res
 
        if res ==2:
        #开始截取音频
            if not audio2:
                audio2 = pyaudio.PyAudio()
                stream2 = audio2.open(format=FORMAT,
                                      channels=1,
                                      rate=22050,
                                      input=True,
                                      frames_per_buffer=256
                                      )
            stream_data = stream2.read(256)
            wave_data = np.frombuffer(stream_data, dtype=np.int16)
            # print(num, wave_data, len(stream_data))
            if speaker!=name_num:
                print(name_num, "正在说话ing...")
                speaker=name_num
            self.frames.append(stream_data)
 
        if res ==3 and len(self.frames)>25:
 
                # print(len(self.frames))
            wf = wave.open("record/"+str(name_num)+".wav", 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(audio2.get_sample_size(FORMAT))
            wf.setframerate(22050)
            wf.writeframes(b"".join(self.frames))
            self.frames = []
            stream2.stop_stream()
            stream2.close()
            audio2.terminate()
            audio2 = ""
            stream2 = ""
            wf.close()
            print(name_num, "结束说话")
            name_num+=1
 
        # if num <=100:
        #     self.x.append(num)
        #     self.y.append(res)
        # elif num > 100:
        #     self.x.append(num)
        #     self.y.append(res)
        #     self.x.pop(0)
        #     self.y.pop(0)
        #
        # plt.cla()
        # # plt.scatter(num, res)
        # plt.plot(self.x, self.y, 'r-', lw=1)
        # plt.pause(0.01)
        # plt.show()
 
        # print("idx={}\t过零率={}\t短时能量={}\tres={}".format(num,zcr,amp,res))
 
    # 短时能量，无声 < 浊音 < 清音
    # 过零率，无声 < 清音 < 浊音
 
    """
    实际测试
    说话时的过零率 在0-3之间    呼吸时的过零率在0-5之间
    说话时的短时能量 在940-12000之间    15000-23000之间
    """
 
    def speech_status(self, amp, zcr):
        status = 0
        # 0= 静音， 1= 可能开始, 2=确定进入语音段   3语音结束
        if self.cur_status in [0, 1]:    #如果在静音状态或可能的语音状态，则执行下面操作
            # 确定进入语音段
            if amp > self.amp1 or zcr > self.zcr1:    #超过最大  短时能量门限了
                status = 2
                self.silence = 0
                self.count += 1
            # 可能处于语音段   能量处于浊音段，过零率在清音或浊音段
            elif amp > self.amp2 or zcr > self.zcr2:
                status = 2
                self.count += 1
            # 静音状态
            else:
                status = 0
                self.count = 0
                self.count = 0
        # 2 = 语音段
        elif self.cur_status == 2:
            # 保持在语音段    能量处于浊音段，过零率在清音或浊音段
            if amp > self.amp2 or zcr > self.zcr2:
                self.count += 1
                status = 2
            # 语音将结束
            else:
                # 静音还不够长，尚未结束
                self.silence += 1
                if self.silence < self.maxsilence:
                    self.count += 1
                    status = 2
                # 语音长度太短认为是噪声
                elif self.count < self.minlen:
                    status = 0
                    self.silence = 0
                    self.count = 0
                # 语音结束
                else:
                    status = 3
                    self.silence = 0
                    self.count = 0
        return status
 
 
 
 
class FileParser(Vad):
    def __init__(self):
        self.block_size = 256
        Vad.__init__(self)
 
 
if __name__ == "__main__":
 
    # plt.ion()
    stream_test = FileParser()
    num = 0
    name_num=1
    speaker=0
    while True:
        byte_obj = stream.read(256)
        stream_test.check_ontime(byte_obj)
        num = num+1

