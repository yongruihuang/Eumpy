# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:10:53 2018

@author: Yongrui Huang
"""

import sys
sys.path.append('../../')
from flask import Flask

import _thread
import time
import configuration

def open_webbrower(thread_name,delay):
    '''
        Open the brower with another thread
        
        Arguments:
        
            thread_name: the name of this thread
            
            delay: how many second do the thread sleep before opening the brower
    '''
    
    time.sleep(delay)
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000/')
    

class RealtimeEmotionDetection():
    
    '''
        This class is used for data collection
        
        Attributes:
        
            app: the flask app
            
            
    '''

    def __init__(self):
        '''
            initialize recorders and bind the app to backend
        '''
        
        
        #creat a Flask obj
        self.app = Flask(__name__)
        
        self.frames = []
        
        self.EEGs = []
        
        import bind

        #bind static html
        bind.bind_html(self)
        
        #bind post request
        bind.bind_post(self)   
        
    
    def start(self):
        '''
            start GUI, first we open the brower then we run the server
        '''
        
        try:
           _thread.start_new_thread(open_webbrower, ("open_webbrower", 0))
        except:
           pass
       
        self.app.run(debug=False, port = configuration.PORT)
       
if __name__ == '__main__':

    realtime_emotion_detection_obj = RealtimeEmotionDetection()
    realtime_emotion_detection_obj.start()
        
    
        
        
        
    
    
    
    
    
    