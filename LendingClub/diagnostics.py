import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import calibration


################################################################################
#                                  Settings                                    #
################################################################################

# numpy display options
np.set_printoptions(precision=6, linewidth=110)

# Show all columns
pd.set_option('display.max_columns', None)

# Show at most 500 rows
pd.set_option('display.max_rows', 500)


################################################################################
#                           Pandas plotting                                    #
################################################################################


# plt.rcParams.update({
#     'figure.figsize': (8, 6),    # figure size in inches
#     #'text.usetex': True,
#     #'font.family': 'serif',
#     #'font.serif': ['Computer Modern'],
#     'font.size': 14,
#     #'axes.titlesize': 20,
#     'axes.labelsize': 20,
#     'axes.linewidth': 2,
#     'xtick.labelsize': 20,
#     'ytick.labelsize': 20,
#     'legend.fontsize': 14,
#     'xtick.major.size': 8,
#     'xtick.minor.size': 4,
#     'xtick.major.width': 2,
#     'xtick.minor.width': 2,
#     'xtick.direction': 'out',
#     'ytick.major.size': 8,
#     'ytick.minor.size': 4,
#     'ytick.major.width': 2,
#     'ytick.minor.width': 2,
#     'ytick.direction': 'out',
#     #'axes.prop_cycle'    : cycler('color', 'bgrcmyk'),
#     })


def plot_category_time_series(df, column, category):
    """Time series curve for the number of rows in column that belong to category.
    """
    gb = df[df[column]==category].groupby('issue_d')
    ax = gb[column].count().plot(label=category)
    return ax


def plot_all_categories_in_column_as_time_series(df, column):
    """
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    categories = df[column].unique()
    for c in categories:
        ax = plot_category_time_series(df, column, c)

    ax.legend(ncol=2)
    ax.set_yscale('log')
    return fig, ax


def bin_prob_y(df, xname, yname, bins):
    """Bin dataframe. Then, calculate fraction of y=1 in each bin.

    Returns
    -------
    ns : 1d-array
        Number in each bin
    ps : 1d-array
        Propability of y=1 in each bin
    ps_err : 1d-array
        Standard error for ps (ps/sqrt(ns)).
    """
    # Just x, y values, and get rid of nans.
    df_xy = df[[xname, yname]].dropna()

    ns = []
    ps = []
    for i in range(len(bins)-1):
        xlow = bins[i]
        xhigh = bins[i+1]
        # Make data frame for bin_i
        df_i = df_xy[(df_xy[xname]>=xlow) & (df_xy[xname]<xhigh)]
        fsurvive = df_i[yname].mean()
        n = len(df_i)
        #print xlow, xhigh, len(df_i), fsurvive
        ns.append(n)
        ps.append(fsurvive)

    return np.array(ns), np.array(ps), np.array(ps)/np.sqrt(np.array(ns))


################################################################################
#                           scikit-learn plotting                              #
################################################################################


def plot_random_forest_feature_importance(clf, X_columns):
    """The importance of each feature.
    See: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    How feature importance is determined:
    https://stackoverflow.com/questions/15810339/how-are-feature-importances-in-randomforestclassifier-determined

    Parameters
    ----------
    clf : RandomForestClassifier object
    X_columns : list of column names
    """
    # Mean value of importance
    importance = clf.feature_importances_
    # DataFrame with each row indexed by the column name X_columns
    importance = pd.DataFrame(importance, index=X_columns, columns=['importance'])
    # Standard deviation of the importance of each tree in the random forest.
    importance['std'] = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

    # Bar plot
    x = range(len(importance))
    y = importance['importance']
    yerr = importance['std']

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.bar(x, y, yerr=yerr, align="center")
    ax.set_xticks(x)
    ax.set_xticklabels(X_columns)
    ax.grid(which='both')
    plt.xticks(rotation=90);

    
def plot_roc_and_threshold(clf, X_true, y_true):
    """Plot a ROC curve (true positive rate vs. false positive rate), and the
    TPR as a function of the threshold to classify an example as a positive.

    TPR = TP/(actual poisitives) = TP/(TP+FN) = recall
    FPR = FP/(actual negatives) = FP/(FP+TN)

    Parameters
    ----------
    clf : Classifier type object that has a .predict_proba() method.
    X_true : 2d array of features. You should use a test set *not* the training set.
    y_true : array of 0s and 1s. You should use a test set *not* the training set.
    """
    # The predicted score (which is roughly a probability)
    y_score = clf.predict_proba(X_true)[:, 1]

    # Calculate ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.plot(fpr, tpr)
    ax1.plot([0, 1], [0, 1], c='k', ls='--')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.minorticks_on()
    ax1.grid(which='major', ls='-')
    ax1.grid(which='minor', ls=':')

    ax2.plot(thresholds, tpr)
    ax2.plot([0.5, 0.5], [0, 1], c='k', ls='--')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Threshold probability')
    ax2.set_ylabel('True Positive Rate')
    ax2.minorticks_on()
    ax2.grid(which='major', ls='-')
    ax2.grid(which='minor', ls=':')


def plot_calibration_curve(fig, ax1, ax2, clf, X_true, y_true, n_bins=20, label=None):
    """Plots the actual probability vs. the predicted probability for a binary classifier.
    Also plots a histogram of the predicted probability.

    Parameters
    ----------
    clf : Classifier type object that has a .predict_proba() method.
    X_true : 2d array of features. You should use a test set *not* the training set.
    y_true : array of 0s and 1s. You should use a test set *not* the training set.
    """
    # Predicted probability of a positive for each example
    y_prob = clf.predict_proba(X_true)[:, 1]

    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration.calibration_curve(y_true, y_prob, n_bins=n_bins)

    # Make the plots
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fig.subplots_adjust(wspace=0, hspace=0)

    ax1.plot(mean_predicted_value, fraction_of_positives, label=label)
    #ax1.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    ax1.plot([0, 1], [0, 1], ls='--', c='k')
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, 1])
    ax1.set_ylabel(r'Actual probability ($N_{\rm positive}/N_{\rm total}$)')
    ax1.minorticks_on()
    ax1.legend()
    ax1.grid(which='major', ls='-')
    ax1.grid(which='minor', ls=':')

    ax2.hist(y_prob, range=(0, 1), bins=n_bins, histtype='step', lw=2)
    ax2.set_xlabel('Predicted probabilty')
    ax2.set_ylabel('Count')
    ax2.set_yscale('log')
    ax2.minorticks_on()
    ax2.grid(which='major', ls='-')
    ax2.grid(which='minor', ls=':')


################################################################################
#                       Useful scikit-learn functions                          #
################################################################################


def classifier_metrics(clf, X, y_true):
    """Useful metrics for evaluating a classifier.

    Parameters
    ----------
    clf : Classifier object
    X : Ground truth features
    y_true : Ground truth labels

    Returns
    -------
    metrics : Dictionary of various metrics
    """
    # Accuracy (fraction of examples classified correctly)
    acc = clf.score(X, y_true)

    y_pred = clf.predict(X)

    tp = np.sum((y_true==1) & (y_pred==1))
    fp = np.sum((y_true==0) & (y_pred==1))
    tn = np.sum((y_true==0) & (y_pred==0))
    fn = np.sum((y_true==1) & (y_pred==0))

    # Precision P = tp / (tp + fp)
    # High precision means there are a small number of false positives
    P = metrics.precision_score(y_true, y_pred, pos_label=1)

    # Recall R = tp / (tp + fn)
    # High recall means there are a small number of false negatives
    # (i.e. you recalled most of the true positives)
    # Also called True Positive Rate / sensitivity
    R = metrics.recall_score(y_true, y_pred, pos_label=1)

    # False Positive Rate
    FPR = fp / (fp + tn)

    # F1 = 2 * (P * R) / (P + R)
    # This is a harmonic (not geometric) mean.
    F1 = metrics.f1_score(y_true, y_pred, pos_label=1)

    # AUROC
    # First probabability is for y=0.
    # Second probability is for y=1. You want the y=1 probability.
    y_prob = clf.predict_proba(X)[:, 1]
    auroc = metrics.roc_auc_score(y_true, y_prob)

    return {'acc':acc, 'P':P, 'R':R, 'F1':F1, 'FPR':FPR, 'auroc':auroc,
            'tp':tp, 'fp':fp, 'tn':tn, 'fn':fn}
