from glob import glob
import os
import pandas as pd
import SimpleITK as sitk
from algorithm import Balaitous
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm


def do_learning(data_dir):

    image_dir = os.path.join(data_dir, "data/mha/")
    metadata = pd.read_csv(os.path.join(data_dir, "metadata/reference.csv"))
    params = {}
    C_range = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 3, 4, 5]
    max_iter = 10000
    n_splits = 8
    n_repeats = 4
    seed = 123
    t_range = np.linspace(0, 1, 21)

    # Create Y
    mask = metadata['probCOVID'].values == 1
    Y = {
        'severe': metadata['probSevere'].values[mask],
        'covid': metadata['probCOVID'].values
    }

    # Get features
    balaitous = Balaitous()

    x_full = []
    x_lung = []
    age = []
    sex = []

    for patiend_id in tqdm(metadata['PatientID']):
        path = os.path.join(image_dir, str(patiend_id) + '.mha')
        image = sitk.ReadImage(path)
        sample = balaitous(image)
        x_full.append(sample['features']['full'])
        x_lung.append(sample['features']['lung'])
        age.append(sample['age'])
        sex.append(sample['sex'])

    # Create X
    X = {'severe': {}, 'covid': {}}
    X['covid']['full'] = np.vstack(x_full)
    X['covid']['lung'] = np.vstack(x_lung)

    age_sex = np.vstack([age, sex]).T
    X['severe']['full'] = np.hstack([X['covid']['full'], age_sex])[mask]
    X['severe']['lung'] = np.hstack([X['covid']['lung'], age_sex])[mask]

    # Get optimal C
    C = {}
    for output in ['severe', 'covid']:
        for k in ['full', 'lung']:
            model = LogisticRegressionCV(Cs=C_range, max_iter=max_iter)
            model.fit(X[output][k], Y[output])
            params[f'C_{k}_{output}'] = model.C_[0]
            params[f'coef_{k}_{output}'] = model.coef_[0]
            params[f'intercept_{k}_{output}'] = model.intercept_[0]

        # Run cross validation
        Y_true = []
        Y_pred = {}
        kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        for train, test in kfold.split(Y[output], Y[output]):
            Y_true.append(Y[output][test])
            for k in ['full', 'lung']:
                C = params[f'C_{k}_{output}']
                model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
                model.fit(X[output][k][train], Y[output][train])
                Y_pred.setdefault(k, []).append(
                    model.predict_proba(X[output][k][test])[:, 1])

        # Get optimal threshold and AUC
        auc = np.zeros((n_repeats * n_splits, len(t_range)))
        for i, (Yt, Yp1, Yp2) in enumerate(zip(Y_true, Y_pred['full'], Y_pred['lung'])):
            auc[i] = [roc_auc_score(Yt, t * Yp1 + (1 - t) * Yp2) for t in t_range]

        at = np.argmax(auc.mean(0))
        t = t_range[at]
        params[f'alpha_full_{output}'] = t
        params[f'alpha_lung_{output}'] = 1 - t

        params[f'auc_full_{output}'] = auc[:, -1]
        params[f'auc_lung_{output}'] = auc[:, 0]
        params[f'auc_{output}'] = auc[:, at]
        params[f'n_{output}'] = len(Y[output])

    print(f'full {params[f"auc_full_severe"].mean()}')
    print(f'lung {params[f"auc_lung_severe"].mean()}')
    print(f'both {params[f"auc_severe"].mean()}')

    np.savez('artifact/balaitous.npz', **params)
    artifacts = glob('artifact/*')
    return artifacts