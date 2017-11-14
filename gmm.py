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


def dev_model(provider):
    data, labels, filenames = provider.get_data('dev')
    mixture0 = _load_mixture(0)
    mixture1 = _load_mixture(1)
    df = pd.DataFrame(columns=['filename', 'label', 'score_0', 'score_1'])
    for i in range(len(data)):
        score0 = mixture0.score(data[i])
        score1 = mixture1.score(data[i])
        df.loc[i] = [filenames[i], labels[i], score0, score1]
    save_file = 'output/dev_result.csv'
    df.to_csv(save_file, index=False)


def eval_model(provider):
    pass


if __name__ == '__main__':
    provider = DataProvider('MFCC')
    train_model(provider)
    dev_model(provider)
    eval_model(provider)
