import json
import os
import librosa
import pandas as pd


def extract_train_features():
    file_dir = 'data/ASVspoof2017_train'
    files = os.listdir(file_dir)
    features = [[], []]
    train_label_file = 'data/labels/ASVspoof2017_train.trn.txt'
    df = pd.read_csv(train_label_file,
                     sep=' ',
                     header=None,
                     index_col=0)
    for i in range(len(files)):
        print(i)
        filename = files[i]
        y, sr = librosa.load(os.path.join(file_dir, filename))
        feature = librosa.feature.mfcc(y=y, sr=sr)
        for y in range(len(feature[0])):
            if df[1][filename] == 'genuine':
                features[0].append(feature[:, y].tolist())
            elif df[1][filename] == 'spoof':
                features[1].append(feature[:, y].tolist())
    with open('output/MFCC/train_data.json', 'w') as fid:
        json.dump(features, fid)


def extract_features():
    phrases = ['train', 'dev', 'eval']
    for phrase in phrases:
        file_dir = 'data/ASVspoof2017_{}'.format(phrase)
        files = os.listdir(file_dir)
        features = []
        for i in range(len(files)):
            print(i)
            filename = files[i]
            y, sr = librosa.load(os.path.join(file_dir, filename))
            feature = librosa.feature.mfcc(y=y, sr=sr)
            feature = [feature[:, x].tolist() for x in range(feature.shape[1])]
            features.append(feature)
        with open('output/MFCC/{}.json'.format(phrase), 'w') as fid:
            json.dump(features, fid)

if __name__ == '__main__':
    extract_features()
