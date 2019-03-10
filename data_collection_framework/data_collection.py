# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:26:22 2018

@author: Yongrui Huang
"""

import sys
import os
o_path = os.path.abspath(__file__)
last_len = len(o_path.split('\\')[-1]) + len(o_path.split('\\')[-2]) + 1
sys.path.append(o_path[:-last_len])
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
    

class DataCollection():
    
    '''
        This class is used for data collection
        
        Attributes:
        
            app: the flask app
            
            records: a list of different recorder object.
    '''

    def __init__(self):
        '''
            initialize recorders and bind the app to backend
        '''
        #initialize recorders list
        self.recorders = []
        
        #creat a Flask obj
        self.app = Flask(__name__)
        
        from data_collection_framework import bind

        #bind static html
        bind.bind_html(self)
        
        #bind post request
        bind.bind_post(self)   
        
    def add_recorder(self, recorder):
        '''
        add one recorder
        
        Arguments:
        
            recorder: one recorder object
        
        '''
        self.recorders.append(recorder)
        
    def add_recorders(self, recorders):
        '''
        add some recorders
        
        Arguments:
        
            recorders: a list that stord some recorders
        '''
        self.recorders = self.recorders + recorders
        
    def start(self):
        '''
            start GUI, first we open the brower then we run the server
        '''
        try:
           _thread.start_new_thread(open_webbrower, ("open_webbrower", 0))
        except:
           pass  
        self.app.run(debug= False, port = configuration.PORT)