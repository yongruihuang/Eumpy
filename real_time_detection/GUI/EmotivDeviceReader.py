# encoding: utf-8
'''
Created on Dec 18, 2018

@author: Yongrui Huang
'''

import time
from array import *
from ctypes import *
from sys import exit
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np

class EmotivDeviceReader(object):
    '''
    classdocs
    This class is used to read EEG data from emotiv
    Attributes:
        queue: the queue save EEG data
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.queue = Queue(maxsize = -1)
        
        
    
    def check_status(self):
        '''
        check if the device is connect correctly, if not, exit this process
        '''
        if self.libEDK.IEE_EngineConnect(create_string_buffer(b"Emotiv Systems-5")) != 0:
            print ("Emotiv Engine start up failed.")
            exit();
        
    def loop(self):
        '''
        the loop is used to continuously read data from device
        '''
        try:
            self.libEDK = cdll.LoadLibrary("win64/edk.dll")
        except Exception as e:
            print ('Error: cannot load EDK lib:', e)
            exit()
        
        self.IEE_EmoEngineEventCreate = self.libEDK.IEE_EmoEngineEventCreate
        self.IEE_EmoEngineEventCreate.restype = c_void_p
        self.eEvent = self.IEE_EmoEngineEventCreate()
        
        self.IEE_EmoEngineEventGetEmoState = self.libEDK.IEE_EmoEngineEventGetEmoState
        self.IEE_EmoEngineEventGetEmoState.argtypes = [c_void_p, c_void_p]
        self.IEE_EmoEngineEventGetEmoState.restype = c_int
        
        self.IEE_EmoStateCreate = self.libEDK.IEE_EmoStateCreate
        self.IEE_EmoStateCreate.restype = c_void_p
        self.eState = self.IEE_EmoStateCreate()
        
        self.IEE_EngineGetNextEvent = self.libEDK.IEE_EngineGetNextEvent
        self.IEE_EngineGetNextEvent.restype = c_int
        self.IEE_EngineGetNextEvent.argtypes = [c_void_p]
        
        self.IEE_EmoEngineEventGetUserId = self.libEDK.IEE_EmoEngineEventGetUserId
        self.IEE_EmoEngineEventGetUserId.restype = c_int
        self.IEE_EmoEngineEventGetUserId.argtypes = [c_void_p , c_void_p]
        
        self.IEE_EmoEngineEventGetType = self.libEDK.IEE_EmoEngineEventGetType
        self.IEE_EmoEngineEventGetType.restype = c_int
        self.IEE_EmoEngineEventGetType.argtypes = [c_void_p]
        
        self.IEE_EmoEngineEventCreate = self.libEDK.IEE_EmoEngineEventCreate
        self.IEE_EmoEngineEventCreate.restype = c_void_p
        
        self.IEE_EmoEngineEventGetEmoState = self.libEDK.IEE_EmoEngineEventGetEmoState
        self.IEE_EmoEngineEventGetEmoState.argtypes = [c_void_p, c_void_p]
        self.IEE_EmoEngineEventGetEmoState.restype = c_int
        
        self.IEE_EmoStateCreate = self.libEDK.IEE_EmoStateCreate
        self.IEE_EmoStateCreate.argtype = c_void_p
        self.IEE_EmoStateCreate.restype = c_void_p
        
        self.IEE_FFTSetWindowingType = self.libEDK.IEE_FFTSetWindowingType
        self.IEE_FFTSetWindowingType.restype = c_int
        self.IEE_FFTSetWindowingType.argtypes = [c_uint, c_void_p]
        
        self.IEE_GetAverageBandPowers = self.libEDK.IEE_GetAverageBandPowers
        self.IEE_GetAverageBandPowers.restype = c_int
        self.IEE_GetAverageBandPowers.argtypes = [c_uint, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
        
        self.IEE_EngineDisconnect = self.libEDK.IEE_EngineDisconnect
        self.IEE_EngineDisconnect.restype = c_int
        self.IEE_EngineDisconnect.argtype = c_void_p
        
        self.IEE_EmoStateFree = self.libEDK.IEE_EmoStateFree
        self.IEE_EmoStateFree.restype = c_int
        self.IEE_EmoStateFree.argtypes = [c_void_p]
        
        self.IEE_EmoEngineEventFree = self.libEDK.IEE_EmoEngineEventFree
        self.IEE_EmoEngineEventFree.restype = c_int
        self.IEE_EmoEngineEventFree.argtypes = [c_void_p]
        
        self.check_status()
        
        userID = c_uint(0)
        user   = pointer(userID)
        ready  = 0
        state  = c_int(0)
        
        alphaValue     = c_double(0)
        low_betaValue  = c_double(0)
        high_betaValue = c_double(0)
        gammaValue     = c_double(0)
        thetaValue     = c_double(0)
        
        alpha     = pointer(alphaValue)
        low_beta  = pointer(low_betaValue)
        high_beta = pointer(high_betaValue)
        gamma     = pointer(gammaValue)
        theta     = pointer(thetaValue)
        channelList = array('I',[3, 7, 9, 12, 16])   # IED_AF3, IED_AF4, IED_T7, IED_T8, IED_Pz 
        
        while (1):
            state = self.IEE_EngineGetNextEvent(self.eEvent)
            
            data = []
            if state == 0:
                eventType = self.IEE_EmoEngineEventGetType(self.eEvent)
                self.IEE_EmoEngineEventGetUserId(self.eEvent, user)
                if eventType == 16:  # libEDK.IEE_Event_enum.IEE_UserAdded
                    ready = 1
                    self.IEE_FFTSetWindowingType(userID, 1);  # 1: libEDK.IEE_WindowingTypes_enum.IEE_HAMMING
                    print ("User added")
                            
                if ready == 1:
                    for i in channelList: 
                        result = c_int(0)
                        result = self.IEE_GetAverageBandPowers(userID, i, theta, alpha, low_beta, high_beta, gamma)
                        
                        if result == 0:    #EDK_OK
#                             print ("%.6f, %.6f, %.6f, %.6f, %.6f \n" % (thetaValue.value, alphaValue.value, 
#                                                                        low_betaValue.value, high_betaValue.value, gammaValue.value))
                            one_read_data = [thetaValue.value, alphaValue.value, low_betaValue.value, high_betaValue.value, gammaValue.value]
                            if len(one_read_data) > 0:
                                data += one_read_data
            elif state != 0x0600:
                print ("Internal error in Emotiv Engine ! ")
            
            if (len(data) > 0):
                self.queue.put(np.array(data))
#             time.sleep(0.01)
            
    def start(self):
        '''
        start a sub-process
        '''
        sub_process = Process(target=self.loop)
        sub_process.start()

    
    def get_data(self):
        '''
        read psd data
        Returns:
        theta, alpha, low_beta, high_beta, gamma in order
        IED_AF3, IED_AF4, IED_T7, IED_T8, IED_Pz in order
        
        '''
        data_list = []
        while self.queue.qsize() > 0:
            ele = self.queue.get()
            data_list.append(ele)
        return data_list
    
    
if __name__ == '__main__':
    device_reader = EmotivDeviceReader()
    device_reader.start()
    time.sleep(5)
    for i in range(5):
        data = device_reader.get_data()
        data = np.array(data)
        print(data)
        time.sleep(1)
    
   
       