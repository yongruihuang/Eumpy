# Eumpy
A emotion data collection and emotion real-time recognition and visualization framework for both user-independence and user-dependence situation.

This framework provide 2 major function.
1. Help researchers conduct emotion stimul experiment like [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/), [MAHNOB-HCI](https://mahnob-db.eu/hci-tagging/)
2. Provide real-time emotion recognition and non-physiological/physiological signal visualization.

## Which signals do we collect and how to collect them?
The framework has applied methods for collectiong human face images and EEG data. But the users can easily add new methods for other signals because we have provided interface for users to imply. As for the hardware, for face images, the default camera will do the job. For EEG data, we use emotiv insight to collect.

## Prerequisites
- Before you start, you should first download the pre-train model from https://pan.baidu.com/s/10pkLaJCSd2N3nz9tJcRXVQ (password:teos) and emotiv support from https://github.com/Emotiv/community-sdk and put it in corresponding position(emotiv is used to collect EEG data) depended on your platform. For example, if you are in windows10 *64 like me, you should download the win64 floder before you put it in data_collection_framework/util/win64 and real_time_detection/GUI.
- [Anaconda (Python 3.7 version)](https://www.anaconda.com/download/#windows)
- [Keras 2.2.4](https://pypi.org/project/Keras/)
- [MNE](https://www.martinos.org/mne/stable/install_mne_python.html)
- Chrome (or other Browser, but chrome is recommended)

## Instruction for data collection
run data_collection_framework/start_GUI.py and the experiment will start.
![image](https://github.com/yongruihuang/Eumpy/blob/master/image/Figure1.png)
<p align="center">Figure 1: the process for data collection</p>

Figure 1 show the process of data collection. To begin with, the subjects were instructed to fill their information. Then the data collection loop begin. After all trials were completed, a finish page would appear. 

The collected data will be saved in dataset floder in root path.

## Instruction for data visualization
run real_time_detection/GUI/start_GUI.py and the process will start.
![image](https://github.com/yongruihuang/Eumpy/blob/master/image/Figure2.png)
<p align="center">Figure 2: the screenshot for data visualization</p>
