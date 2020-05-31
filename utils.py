from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm_notebook
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from bert_cls import *
from bert_aug import *

def base_experiments(X, y, n_epochs, lrs, random_states, n_folds):
    exp_results = {}
    for random_state in random_states:
        for lr in lrs:
            for n_epoch in n_epochs:
                params = 'state = '+str(random_state) +'lr = ' \
                + str(lr) + 'epochs' + str(n_epoch)
                pred, y_true = get_metrics_base(X, y, random_state, lr, n_epoch,
                n_folds)
                acc = accurancy_score(pred, y_true)
                exp_results[params] = acc
    return exp_results


def accurancy_score(p, t):
  res = []
  for i ,p_i in enumerate(p):
    n = len(p_i)
    tr = 0
    for j, p_val in enumerate(p_i):
      if t[i][j] == p[i][j]:
        tr += 1
    res.append(tr/n)
  return res


def get_metrics_base(X, y, random_state, lr, n_epochs, n_folds=2):
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    y_tests = []
    pred = []
    er_list = []
    for train_index, test_index in tqdm_notebook(kf.split(X, y)):

        model = BertClassificationModel(n_epochs=n_epochs, lr=lr)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train,y_train)

        predict = model.predict_batch_proba(X_test)
        y_tests.append(y_test)

        predict_flat = []
        for pr in predict:
          pr_lbl = np.argmax(pr, axis=1).tolist()
          predict_flat += pr_lbl

        pred.append(predict_flat)

    return pred, y_tests

def accurancy_score(p, t):
  res = []
  for i ,p_i in enumerate(p):
    n = len(p_i)
    tr = 0
    for j, p_val in enumerate(p_i):
      if t[i][j] == p[i][j]:
        tr += 1
    res.append(tr/n)
  return res


def fit_predict(X, y, X_test, n_epochs,lr):
    model = BertClassificationModel(n_epochs=n_epochs, lr=lr)
    model.fit(X,y)
    predict = model.predict_batch_proba(X_test)
    predict_flat = []
    for pr in predict:
        pr_lbl = np.argmax(pr, axis=1).tolist()
        predict_flat += pr_lbl
    return predict_flat


def get_metrics_aug(X,y ,n_folds=2, temp=1, n_random=2, n_epochs=4,
                    if_save_test=True, if_aug_neaded=True, if_save_train=True,
                    random_state = None, use_untuned=False, if_lbl=True,
                    use_stop = False, lr=2e-5, if_orig_needed=True,
                    n_samples=4, n_rounds=2, d_lbl=None):
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    aug_bert = BertAug(model_name='imdb_fine/5', use_untuned=use_untuned,
                       use_stop = use_stop)

    y_tests = []
    pred_orig = []
    pred_aug = []
    er_list = []
    saved_test = []
    auged_train = []

    for train_index, test_index in tqdm_notebook(kf.split(X, y)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if if_save_test:
          saved_test.append(X_test)

        y_tests.append(y_test)

        if if_orig_needed:

          predict_flat_orig = fit_predict(X_train, y_train, X_test,
                                          n_epochs=n_epochs, lr=lr)
          pred_orig.append(predict_flat_orig)

        if if_aug_neaded:

          X_train_aug = []
          y_train_aug = []

          for i, sent in enumerate(X_train):
            if if_lbl and d_lbl[X_train[i]] == 0 or d_lbl[X_train[i]] == 1:
              if d_lbl[X_train[i]] == 0:
                new_sents = aug_bert.aug_sent(sent, label = 0, temp=temp,
                                          n_random=n_random,
                                          n_samples=n_samples*2,
                                          n_rounds=n_rounds)
                X_train_aug += new_sents
                y_train_aug += [y_train[i]] * len(new_sents)
              elif d_lbl[X_train[i]] == 1:
                new_sents = aug_bert.aug_sent(sent, label = 1, temp=temp,
                                          n_random=n_random,
                                          n_samples=n_samples*2,
                                          n_rounds=n_rounds)
                X_train_aug += new_sents
                y_train_aug += [y_train[i]] * len(new_sents)
              if if_save_train:
                auged_train.append(X_train_aug)
            else:
              new_sents1 = aug_bert.aug_sent(sent, label = 0, temp=temp,
                                            n_random=n_random,
                                             n_samples=n_samples,
                                             n_rounds=n_rounds)
              X_train_aug += new_sents1
              y_train_aug += [y_train[i]] * len(new_sents1)

              if if_save_train:
                auged_train.append(X_train_aug)

              new_sents2 = aug_bert.aug_sent(sent, label = 1, temp=temp,
                                            n_random=n_random,
                                             n_samples=n_samples,
                                             n_rounds=n_rounds)
              X_train_aug += new_sents2[1:]
              y_train_aug += [y_train[i]] * (len(new_sents2)-1)

          X_train_aug = np.array(X_train_aug)
          y_train_aug = np.array(y_train_aug)

          predict_flat_aug = fit_predict(X_train_aug, y_train_aug, X_test,
                                         n_epochs=n_epochs, lr=lr)
          pred_aug.append(predict_flat_aug)

    return pred_orig, pred_aug, y_tests, saved_test, auged_train



def drow_plots(po, pa):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))

    x = np.arange(1, len(po)+1)

    ax1.set_xticks(np.linspace(0, len(po), len(po)+1))
    ax1.set_yticks(np.arange(0, 1, step=0.05))

    ax1.plot(x, po,  color='red', label='без аугментации')
    ax1.plot(x, pa,  color='green', label='с аугментацией')

    ax1.set_xlabel('разбиение')
    ax1.set_ylabel('точность(accuracy)')

    ax1.grid()
    ax1.legend()

    boxplot_data = [po, pa]
    boxplot_labels = ['без аугментации', 'с аугментацией']

    ax2.boxplot(boxplot_data, labels=boxplot_labels)
    ax2.set_ylabel('точность(accuracy)')
    ax2.grid()

    plt.show()


def aug_experiments(X, y, n_epochs, lrs, random_states, n_folds, use_stop_vals,
                    temps, n_random_vals):
    exp_results = {}
    for random_state in random_states:
        for lr in lrs:
            for n_epoch in n_epochs:
                for temp in temps:
                    for n_random in n_random_vals:
                        for use_stop in use_stop_vals:
                            params = (random_state, lr, n_epoch,
                                temp, n_random, use_stop)
                            base, aug, y_true, _, _ = get_metrics_aug(X, y,
                            n_folds=n_folds, n_random=n_random, n_epochs=n_epoch,
                            if_save_test=False, if_aug_neaded=True,
                            if_save_train=False,random_state = None,
                            use_untuned=False, if_lbl=True, use_stop = use_stop,
                            lr=lr, if_orig_needed=False)
                            acc = accurancy_score(aug, y_true)
                            exp_results[params] = acc
    return exp_results
