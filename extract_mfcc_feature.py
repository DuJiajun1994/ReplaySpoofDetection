import json
import os
import librosa
import pandas as pd

train_label_file = 'ASVspoof2017_train.trn.txt'
dev_label_file = 'ASVspoof2017_dev.trl.txt'
eval_label_file = 'ASVspoof2017_eval_v2_key.trl.txt'


def _get_train_features(phase):
    file_dir = 'data/ASVspoof2017_{}'.format(phase)
    if phase == 'train':
        label_file = train_label_file
    elif phase == 'dev':
        label_file = dev_label_file
    files = os.listdir(file_dir)
    features = [[], []]
    df = pd.read_csv(os.path.join('data/labels', label_file),
                     sep=' ',
                     header=None,
                     index_col=0)
    for i in range(len(files)):
        filename = files[i]
        print('{} {}'.format(i, filename))
        y, sr = librosa.load(os.path.join(file_dir, filename))
        feature = librosa.feature.mfcc(y=y, sr=sr)
        for y in range(len(feature[0])):
            if df[1][filename] == 'genuine':
                features[0].append(feature[:, y].tolist())
            elif df[1][filename] == 'spoof':
                features[1].append(feature[:, y].tolist())
    return features


def extract_train_features():
    train_features = _get_train_features('train')
    dev_features = _get_train_features('dev')
    train_dev_features = [train_features[0] + dev_features[0], train_features[1] + dev_features[1]]
    with open('output/MFCC/train/train.json', 'w') as fid:
        json.dump(train_features, fid)
    with open('output/MFCC/train/train_dev.json', 'w') as fid:
        json.dump(train_dev_features, fid)


def extract_eval_features():
    phases = ['train', 'dev', 'eval']
    for phase in phases:
        if phase == 'train':
            label_file = train_label_file
        elif phase == 'dev':
            label_file = dev_label_file
        elif phase == 'eval':
            label_file = eval_label_file
        df = pd.read_csv(os.path.join('data/labels', label_file),
                         sep=' ',
                         header=None,
                         index_col=0)
        file_dir = 'data/ASVspoof2017_{}'.format(phase)
        files = list(df.index)
        features = []
        for i in range(len(files)):
            filename = files[i]
            print('{} {}'.format(i, filename))
            y, sr = librosa.load(os.path.join(file_dir, filename))
            feature = librosa.feature.mfcc(y=y, sr=sr)
            feature = [feature[:, x].tolist() for x in range(feature.shape[1])]
            features.append(feature)
        with open('output/MFCC/eval/{}.json'.format(phase), 'w') as fid:
            json.dump(features, fid)

if __name__ == '__main__':
    extract_train_features()
    extract_eval_features()
