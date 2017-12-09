import pickle
import pandas as pd
from sklearn.mixture import GaussianMixture
from data_provider import DataProvider

num_components = 512


def _train_mixture(x, label):
    mixture = GaussianMixture(n_components=num_components)
    mixture.fit(x)
    with open('output/mixture{}_params.pkl'.format(label), 'wb') as fid:
        pickle.dump(mixture, fid)


def _load_mixture(label):
    with open('output/mixture{}_params.pkl'.format(label), 'rb') as fid:
        mixture = pickle.load(fid)
    return mixture


def train_model(provider):
    data = provider.get_train_data()
    _train_mixture(data[0], 0)
    _train_mixture(data[1], 1)


def eval_model(provider, phase):
    assert phase in ['train', 'dev', 'eval']
    data, labels, files = provider.get_data(phase)
    mixture0 = _load_mixture(0)
    mixture1 = _load_mixture(1)
    df = pd.DataFrame(columns=['filename', 'score'])
    for i in range(len(data)):
        print(i)
        genuine_score = mixture0.score(data[i])
        spoof_score = mixture1.score(data[i])
        df.loc[i] = [files[i], spoof_score - genuine_score]
    save_file = 'output/{}_result.csv'.format(phase)
    df.to_csv(save_file, sep=' ', header=False, index=False)


if __name__ == '__main__':
    provider = DataProvider('MFCC')
    train_model(provider)
    eval_model(provider, 'train')
    eval_model(provider, 'dev')
    eval_model(provider, 'eval')
