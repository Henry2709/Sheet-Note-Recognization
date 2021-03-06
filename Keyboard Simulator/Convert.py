#!/usr/bin/env python
'''
Keyboard Simulator.

Given input, it can compile the music and output a WAV file.
'''
import pysynth_b as ps 
import pyaudio  
import wave  

test = (('c', 4),('d', 4), ('e', 4), ('f', 4),('g', 4),('a', 4),('b', 4),('c5', 4),
        ('d5', 4),('e5', 4),('d', 4),('e', 4),('e', -4),('d', 8),('d', 2),
        ('e', 4),('e', 4), ('f', 4), ('g', 4),('g', 4),('f', 4),('e', 4),('d', 4),
        ('c', 4),('c', 4),('d', 4),('e', 4),('d', -4),('c', 8),('c', 2),
        ('d', 4),('d', 4),('e', 4),('c', 4),('d', 4),('e', 8),('f', 8),('e', 4),('c', 4),
        ('d', 4),('e', 8),('f', 8),('e', 4),('d', 4),('c', 4),('d', 4),('g3', 4),
        ('e', 4),('e', 4), ('f', 4), ('g', 4),('g', 4),('f', 4),('e', 4),('d', 4),
        ('c', 4),('c', 4),('d', 4),('e', 4),('d', -2),('c', 8),('c', 2))

ps.make_wav(test, fn = "test.wav")


chunk = 1024  

f = wave.open("test.wav","rb")  
#instantiate PyAudio  
p = pyaudio.PyAudio()  
#open stream  
stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                channels = f.getnchannels(),  
                rate = f.getframerate(),  
                output = True)  
#read data  
data = f.readframes(chunk)  
  
#paly stream  
while data != '':  
    stream.write(data)  
    data = f.readframes(chunk)  
  
#stop stream  
stream.stop_stream()  
stream.close()  
  
#close PyAudio  
p.terminate()  