# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:39:49 2018

@author: Yongrui Huang
"""

import sys
sys.path.append('../../')
sys.path.append('../')
import configuration

import cv2
import os
import random
import string
import queue
import numpy as np
import FaceFeatureReader

import tensorflow as tf
graph = tf.get_default_graph()

face_feature_reader_obj = FaceFeatureReader.FaceFeatureReader(graph)

class FaceReader:
    '''
    This class is used to return the face data in real time.
   
    Attribute:
        cap: the capture stream
        faceCascade: model for detecting where the face is.
        file_name: the file name of the current frame in hard disk
        delete_queue: the queue is used to save all the delete file name
        faces: the faces for predicting the emotion, we used a set of face 
        rather than one face.
    '''
    
    def __init__(self, input_type, file_path = None):
        '''
        Arguments:
            input_type: 'file' indicates that the stream is from file. In other
            case, the stream will from the defalt camera.
        '''
        self.input_type = input_type
        if input_type == 'file':
            self.cap = cv2.VideoCapture(file_path)
        else:
            self.cap = cv2.VideoCapture(0)
            ret, frame = self.cap.read()
            
        cascPath = configuration.MODEL_PATH + "haarcascade_frontalface_alt.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath) 
        self.file_name = None
        self.delete_queue = queue.Queue()
        self.faces = []
        
        
    
    def delete_files(self):
        '''
        delete files for releasing the resourse.
        '''
        while self.delete_queue.qsize() > 10:
            file = self.delete_queue.get()
            if (os.path.exists(file)):
                os.remove(file)
    
    def get_one_face(self):
        '''
        Returns:
            one face from stream.
        '''
        if self.input_type == 'file':
            cnt = 0
            while cnt < 15:
                self.cap.read()
                cnt += 1
        ret, frame = self.cap.read()
        if ret is True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (x, y, w, h) = self.detect_face(gray)
            if (w != 0):   
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                self.faces.append(face)
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), thickness = 2)
            if self.file_name is not None:
                del_file_name = 'static/cache_image/%s.png'%self.file_name
                self.delete_queue.put(del_file_name)
                
        
                if self.delete_queue.qsize() > 50:
                    self.delete_files()
            
            self.file_name = ''.join(random.sample(string.ascii_letters + string.digits, 12))
            
            cv2.imwrite('static/cache_image/%s.png'%self.file_name, frame)
            
            return self.file_name
        else:
            return 'ERROR'
        
    def detect_face(self, gray):
        '''
        find faces from a gray image.
        Arguments:
            gray: a gray image
        Returns:
            (x, y, w, h)
            x, y: the left-up points of the face
            w, h: the width and height of the face
        '''
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize=(32, 32)
        )
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
        else:
            (x, y, w, h) = (0, 0, 0, 0)
            
        return (x, y, w, h)
    
    def read_face_feature(self):
        '''
        Returns:
            items: a list, the first element is the frame path while the rest 
            is the feature map.
        '''
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = self.detect_face(gray)
        if (w != 0):   
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face_feature_reader_obj.set_face(face)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), thickness = 2)
        random_str = ''.join(random.sample(string.ascii_letters + string.digits, 12))
        frame_path = 'static/cache_image/%s.png'%random_str
        cv2.imwrite(frame_path, frame)
        
        feature_map_list = face_feature_reader_obj.read_feature_map()
        items = [frame_path,]
        items += feature_map_list
        
        self.delete_queue.put(frame_path)
        if self.delete_queue.qsize() > 10:
            self.delete_files()
        return items

import mne
from real_time_detection.EEG import EEG_feature_extract
from EmotivDeviceReader import EmotivDeviceReader

emotiv_reader = EmotivDeviceReader()
emotiv_reader.start()

class EEGReader:
    '''
    This class is used to return the EEG data in real time.
    Attribute:
        raw_EEG_obj: the data for file input. MNE object.
        timestamp: the current time. how much second.
        features: the EEG features.        
    '''
    def __init__(self, input_type, file_path = None):
        '''
        Arguments:
            input_type: 'file' indicates that the stream is from file. In other
            case, the stream will from the 'Emotiv insight' device.
        '''
        self.input_type = input_type
        if self.input_type == 'file':
            self.raw_EEG_obj = mne.io.read_raw_fif(file_path, preload=True)
            max_time = self.raw_EEG_obj.times.max()
            self.raw_EEG_obj.crop(28, max_time)
            self.raw_EEG_obj
            self.timestamp = 0.
            
            cal_raw = self.raw_EEG_obj.copy()
            cal_raw = EEG_feature_extract.add_asymmetric(cal_raw)
            
            self.features = EEG_feature_extract.extract_average_psd_from_a_trial(cal_raw, 1, 0.)
        else:
            #TODO: read EEG from devic
            self.timestamp = 0.
            pass
    
    def get_EEG_data(self):
        '''
        Return:
            EEG data: the EEG data
            timestamp: the current timestamp
        '''
        if self.input_type == 'file':
            sub_raw_obj = self.raw_EEG_obj.copy().crop(self.timestamp, 
                                               self.timestamp + 1.)
            self.timestamp += 1.
            
            show_raw_obj = sub_raw_obj.copy().pick_channels(['AF3','AF4','T7','T8','Pz'])
            
            return show_raw_obj.get_data(), self.timestamp-1., None
        else:
            self.timestamp += 1.
            data_list = emotiv_reader.get_data()
            PSD_feature = np.array(data_list)
            PSD_feature = PSD_feature.reshape(PSD_feature.shape[0], 5, -1)
#             raw_EEG = np.mean(PSD_feature, axis = 2)
            raw_EEG = PSD_feature[:,:,4]
            raw_EEG_fill = np.zeros((257, 5))
            for i in range(raw_EEG.shape[0]):
                start = i*(int(257/raw_EEG.shape[0]))
                end = (i+1)*(int(257/raw_EEG.shape[0])) if i != raw_EEG.shape[0]-1 else raw_EEG_fill.shape[0]
                raw_EEG_fill[start:end, :] = raw_EEG[i]
            return raw_EEG_fill.T, self.timestamp-1., PSD_feature
            
import keras
def format_raw_images_data(imgs, X_mean):
    '''
        conduct normalize and shape the image data in order to feed it directly
        to keras model
       
        Arguments:
        
            imgs: shape(?, 48, 48), all pixels are range from 0 to 255
            
            X_mean: shape: (48, 48), the mean of every feature in faces
        
        Return:
        
            shape(?, 48, 48, 1), image data after normalizing
        
    '''
    imgs = np.array(imgs) - X_mean
    return imgs.reshape(imgs.shape[0], 48, 48, 1)

class EmotionReader:
    '''
    This class is used to return the emotion in real time.
    Attribute:
        input_tpye: input_type: 'file' indicates that the stream is from file.
        In other case, the stream will from the default camera.
        face_model: the model for predicting emotion by faces.
        EEG_model: the model for predicting emotion by EEG.
        todiscrete_model: the model for transforming continuous emotion (valence
        and arousal) into discrete emotion.
        face_mean: the mean matrix for normalizing faces data.
        EEG_mean: the mean matrix for normalizing EEG data.
        EEG_std: the std matrix for normalizing EEG data.
        valence_weigth: the valence weight for fusion
        aoursal_weight: the arousal weight for fusion
        cache_valence: the most recent valence, in case we don't have data to 
        predict we return the recent data.
        cacha_arousal: the most recent arousal.
    
    '''
    def __init__(self, input_type):
        '''
        Arguments:
            input_type: 'file' indicates that the stream is from file. In other
            case, the stream will from the defalt camera.
        '''
        self.input_type = input_type
        
        self.face_model = keras.models.load_model(configuration.MODEL_PATH + 'CNN_face_regression.h5')
        self.EEG_model = keras.models.load_model(configuration.MODEL_PATH + 'LSTM_EEG_regression.h5')
        self.todiscrete_model = keras.models.load_model(configuration.MODEL_PATH + 'continuous_to_discrete.h5')
        self.face_mean = np.load(configuration.DATASET_PATH + 'fer2013/X_mean.npy')
        self.EEG_mean = np.load(configuration.MODEL_PATH + 'EEG_mean.npy')
        self.EEG_std = np.load(configuration.MODEL_PATH + 'EEG_std.npy')
        (self.valence_weight, self.arousal_weight) = np.load(configuration.MODEL_PATH + 'enum_weights.npy')
        
        self.cache_valence, self.cache_arousal = None, None
        self.cnt = 0
        
        if self.input_type == 'file':
            pass
        else:
            #TODO:1
            pass
    
    def get_face_emotion(self):
        '''
        Returns: 
            valence: the valence predicted by faces
            arousal: the arousal predicted by faces
        '''
        
        features = np.array(face_reader_obj.faces)
        if len(features) == 0:
            return None, None
        features = format_raw_images_data(features, self.face_mean)

        with graph.as_default():
            (valence_scores, arousal_scores) = self.face_model.predict(features)
        face_reader_obj.faces = []
        return valence_scores.mean(), arousal_scores.mean()
    
    def get_EEG_emotion(self):
        '''
        Returns:
            valence: the valence predicted by EEG
            arousal: the arousal predicted by EEG
        '''
        
        X = EEG_reader_obj.features[self.cnt-10:self.cnt]
        X = np.array([X, ])
        X -= self.EEG_mean
        X /= self.EEG_std
        print (X.shape)
        with graph.as_default():
            (valence_scores, arousal_scores) = self.EEG_model.predict(X)
        return valence_scores[0][0], valence_scores[0][0]
        
    def get_continuous_emotion_data(self):
        '''
        Returns:
            valence: the valence value predicted by final model
            arousal: the arousal value predicted by final model
        '''
        face_valence, face_arousal = self.get_face_emotion()
        if face_valence is None:
            face_valence = self.cache_valence
            face_arousal = self.cache_arousal
        if self.cnt < 10 or self.input_type != 'file':
            self.cache_valence = face_valence
            self.cache_arousal = face_arousal
            
            return face_valence, face_arousal
            
        
        EEG_valence, EEG_arousal = self.get_EEG_emotion()
        
        valence = self.valence_weight*face_valence + (1-self.valence_weight)*EEG_valence
        arousal = self.arousal_weight*face_arousal + (1-self.arousal_weight)*EEG_arousal
        
        self.cache_valence = valence
        self.cache_arousal = arousal
        
        return valence, arousal
    
    def get_emotion_data(self):
        '''
        Returns:
            cnt: the timestamp
            valence: the valence value predicted by final model.
            arousal: the arousal value predicted by final model.
            discrete_emotion: a vector contains 32 emotion scores.
            emotion_strength: the emotion strength.
        '''
        valence, arousal = self.get_continuous_emotion_data()
        
        X = np.array([[valence, arousal],])
        with graph.as_default():
            distcrte_emotion, emotion_strength = self.todiscrete_model.predict(X)
        self.cnt += 1 
        
        return self.cnt, valence, arousal, distcrte_emotion[0], emotion_strength[0][0]

trial_path = configuration.DATASET_PATH + 'MAHNOB_HCI/18/trial_1/'
# face_reader_obj = FaceReader('file', trial_path + 'video.avi')
# EEG_reader_obj = EEGReader('file', trial_path + 'EEG.raw.fif')
# emotion_reader_obj = EmotionReader('file')
face_reader_obj = FaceReader(input_type='')
EEG_reader_obj = EEGReader(input_type='')
emotion_reader_obj = EmotionReader(input_type='')


if __name__ == '__main__':
   
    for i in range(5):
        for j in range(25):
            face_reader_obj.get_one_face()
        EEG_reader_obj.get_EEG_data()
        print (emotion_reader_obj.get_emotion_data())

     