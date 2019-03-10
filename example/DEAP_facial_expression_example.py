# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:12:36 2018

@author: Yongrui Huang

  This is an example for processing facial expression data in DEAP dataset using the function in ../algorithm_implement/face_tool script
  Note that the dataset's format should be same as the format performed in 'dataset' folder
  The DEAP dataset contains 40 trials for each subject.
  In this example, for each subject, 20 trials are for training whereas the other trials are for testing. 
"""

import sys
sys.path.append('../algorithm_implement')
sys.path.append('..')
from algorithm_implement import face_tool
import configuration

if __name__ == '__main__':
    
    root_path = configuration.DATASET_PATH + 'DEAP/'
    
    for subject_id in range(1, 10):
        
        subject_path = root_path + str(subject_id) + '/'
        face_model = face_tool.FaceModel()
        
        #use 20 trial as train set
        for train_trial_id in range(1, 21):
            
            #load train data
            path = subject_path + 'trial_' + str(train_trial_id) + '/'
            face_model.add_one_trial_data(path)
            
        face_model.train()
        
        #calculate accuracy
        acc_valence, acc_arousal = 0, 0
        
        #other 20 for testing
        for test_trial_id in range(21, 41):
            path = subject_path + 'trial_' + str(test_trial_id) + '/'
            valence_correct, arousal_correct = face_model.predict_one_trial(path)
            if(valence_correct):
                acc_valence+=1
            if(arousal_correct):
                acc_arousal+=1
                
        print ('subject: ' + str(subject_id))
        print (acc_valence/20., acc_arousal/20.)
            
            
       
            