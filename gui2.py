#Program for GUI for activity detection in real time
from tkinter import *
import os
import subprocess
import alsaaudio
import numpy as np
import scipy
import soundfile as sf
import sklearn
import scipy.fftpack
import sklearn.linear_model as sk
import sklearn.svm as svk

#Obtaining the fitted random forest model from train_real.py
from train_real import *

import threading
import time
import subprocess
import queue as Queue

# Function to compute FFT and power spectrum
def fftfunc(fr,nfft):
    l=np.fft.rfft(fr,nfft)
    l=np.absolute(l)
    l=(1/2048)*(np.square(l))
    return l

def start():
    #computing frame size and number of frames
    f=48000
    overlap=int(f*0.01)
    no_frames=int(480000/overlap)
    framesize=int(f*0.025)
    
    #Obtaining 20 mel banks
    print('Calculating mel banks')
    lowfreq=0
    highfreq=2595*np.log10(1+24000/700.)

    melpoints = np.linspace(lowfreq,highfreq,22)
    
    melpoints1=700*(10**(melpoints/2595.0)-1)

    bins=np.floor((2049)*melpoints1/48000)

    fbank = np.zeros((20,1025))
    for j in range(0,20):
        for i in range(int(bins[j]), int(bins[j+1])):
            fbank[j,i] = (i - bins[j]) / (bins[j+1]-bins[j])
        for i in range(int(bins[j+1]), int(bins[j+2])):
            fbank[j,i] = (bins[j+2]-i) / (bins[j+2]-bins[j+1])

    #Initialising variables for MFCC ,Delta and D-delta computations
    frames=np.zeros((no_frames,framesize))
    comp_spec=np.ndarray((no_frames,1025))
    feat1=np.ndarray((1000,20))
    feat=np.ndarray((1000,13))
    pad=np.zeros((1000,17))
    delta=np.zeros((1000,13))
    dpad=np.zeros((1000,17))
    ddelta=np.zeros((1000,13))
    #Thread one to record data and store in queue
    def Threadone(s,q):
        while True:
            subprocess.run(["arecord","--format=S16_LE","--rate=48000","--channels=1","-d","10","threadreal.wav"])
            s,f=sf.read('threadreal.wav')
            q.put(s)

    #Thread two to obtain data from queue and predict 
    def Threadtwo(s,q):
        while True:
            s=q.get()         
            featvect=np.zeros((1000,39))

            #Storing the audio into frames
            for i in range(0,no_frames):
                for j in range(0,framesize):
                    t=i*overlap+j
                    if t<len(s):
                        frames[i][j]=s[t]
            #Calling the power spectra function           
            comp_spec=fftfunc(frames,2048)

            #Calculating the MFCC
            feat1=np.dot(comp_spec,fbank.T)
            feat1=np.where(feat1==0,np.finfo(float).eps,feat1)
            feat1=np.log(feat1)
            feat1=scipy.fftpack.dct(feat1,norm='ortho')
                
            feat=feat1[:,:13]
                
            #Calculating the Delta coefficients                
            for i in range(0,1000):
                for j in range(2,15):
                    pad[i][j]=feat[i][j-2]
            for i in range(0,1000):
                for j in range(0,13):
                    delta[i][j]=np.dot(np.arange(-2,3),pad[i][j:j+5])/10

            #Calculating the d-Delta coefficients
            for i in range(0,1000):
                for j in range(2,15):
                    dpad[i][j]=pad[i][j-2]
            for i in range(0,1000):
                for j in range(0,13):
                    ddelta[i][j]=np.dot(np.arange(-2,3),dpad[i][j:j+5])/10

            #Storing the 3 features as a single array
            for i in range(0,1000):
                for j in range(0,13):
                    featvect[i][j]=feat[i][j]
                    featvect[i][j+13]=delta[i][j]
                    featvect[i][j+26]=ddelta[i][j]
            featvect=featvect.reshape(1,39000)

            #Predicting the feature
            y_pred=clf1.predict(featvect)
            if y_pred==0:
                print("Typing")
            elif y_pred==1:
                print("Eating")
            elif y_pred==2:
                print("Hairdrying")
            elif y_pred==3:
                print("Laundry")
            elif y_pred==4:
                print("Vacuuming")
    #Main function to call the threads
    def main():
        q=Queue.Queue()
        s=np.zeros((48000*10))
        threading.Thread(target=Threadone,args = (s,q)).start()
        threading.Thread(target=Threadtwo,args=(s,q)).start()
    if __name__=='__main__':main()

#Function to print on text box
def redirector(inputStr):
    textbox.insert(INSERT,inputStr)
sys.stdout.write=redirector
master=Tk()

#Creating button and textbox for the GUI
b1=Button(master,text='Start', command=start)
b1.pack()
textbox=Text(master)
textbox.pack()
master.mainloop()
