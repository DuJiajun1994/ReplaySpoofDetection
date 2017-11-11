import numpy as np
import os
import librosa

phrases = ['train', 'dev', 'eval']
for phrase in phrases:
    file_dir = 'data/ASVspoof2017_{}'.format(phrase)
    files = os.listdir(file_dir)
    features = []
    for x in files:
        y, sr = librosa.load(os.path.join(file_dir, x))
        feature = librosa.feature.mfcc(y=y, sr=sr)
        features.append(feature)
    np.save('data/{}_features.npy'.format(phrase), features)
