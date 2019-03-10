'''
Created on 2017/11/3

@author: hyr
'''

import pandas as pd
import numpy as np
import os
import configuration

def makedir():
    '''
        Create a subject's directory in target path from configuration.py
    '''
    base_path = configuration.COLLECT_DATA_PATH
    i = 0
    while True:
        i += 1
        path =  '%s%d/' % (base_path ,i,)
        
        is_exists=os.path.exists(path)
        if not is_exists:
            os.makedirs(path)
            break
    return i, path

def make_trial_dir(path, trial_num):
    '''
        Create trial's directory for subjects
        
        Arguments:
        
            path: the target path
            
            trial_num: how much trail this subject will do.
    '''
    for i in range(1, int(trial_num)+1):
        os.makedirs(path + "/trial_" + str(i))
    
def creat_subject(name, age, gender, trial_num):
    '''
        Create a subject csv file for each subject when they fill their information
       
        Arguments:
        
            name: the subject's name in English, a string
            
            age: the subject's age, a interger
            
            gender: the subject's gender, female or male
            
            trial_num: the number of trial subject intend to do, ranges from 1 to 40
    '''
    subject_id, path = makedir()
    information_frame = pd.DataFrame(np.array([[subject_id, name, age, gender],]), columns=['subject_id', 'name', 'age', 'gender'])
    information_frame.to_csv(path+'/information.csv', index = False)
    make_trial_dir(path, trial_num)
    subject_id = int(subject_id)
    np.save("subject_id.npy",subject_id)

def save_SAM_label(trial_id, valence, arousal):
    '''
        Creat a csv file to storage the ground truth lable of the subject
        
        Arguments:
        
            trial_id: which trial it is.
            
            valence: the ground truth label in valence space, between 1 and 9 
            
            arousal: the ground truth label in arousal space, between 1 and 9
    '''
    
    base_path = configuration.COLLECT_DATA_PATH
    subject_id = np.load('subject_id.npy')
    path = base_path + str(subject_id) + '/trial_' + str(trial_id) + '/label.csv'
    SAM_dataframe = pd.DataFrame(np.array([[valence, arousal],]),columns=['valence', 'arousal'])
    SAM_dataframe.to_csv(path, index=False)
    
if __name__ == '__main__':
    pass