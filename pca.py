import matplotlib.pyplot as plt

import preprocess as pr
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D


def pca_lda(X, y, N):
    pca = PCA(n_components=N)
    d1 = pca.fit(X).transform(X)

    '''
    lda = LinearDiscriminantAnalysis(n_components=N)
    d2 = lda.fit_transform(X, y)

    '''

    # Percentage of variance explained for each components
    print('explained variance ratio (first three components): %s'
      % str(pca.explained_variance_ratio_))


    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    d1 = PCA(n_components=N).fit_transform(X)
    ax.scatter(d1[:, 0], d1[:, 1], d1[:, 2], c=y, cmap=plt.cm.Paired)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    '''
    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    labels = ['Perished', 'Survived']
    bins = np.linspace(-4, 4, 50)
    for color, i, target_name in zip(colors, [0, 1, 2], labels):
        plt.hist(d2[y == i,0], bins, alpha=0.5)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of train dataset')

    '''
    plt.show()
    return d1

if __name__ == '__main__':
    train_path = pr.process_data('train.csv')
    train_df = pd.DataFrame.from_csv(train_path)
    train_data = train_df.values
    test_path = pr.process_data('test.csv')
    test_df = pd.DataFrame.from_csv(test_path)
    test_data = test_df.values
    test_label = np.hstack(pd.DataFrame.from_csv('gender_submission.csv').values)
    N = 3

    '''
    train_X = train_data[0:,1:]
    train_y = train_data[0:,0]
    pca_lda(train_X, train_y, N)

    '''

    test_X = test_data[0:,0:]
    test_y = test_label
    data = pca_lda(test_X, test_y, N)
