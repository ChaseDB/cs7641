from sklearn.tree import DecisionTreeClassifier, export_graphviz
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import learning_curve, cross_val_score, train_test_split, cross_val_predict, StratifiedKFold, cross_validate, GridSearchCV, ShuffleSplit
import pydot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score, average_precision_score, roc_curve, accuracy_score, make_scorer, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import time

# the following line might need to be commented out on some machines
matplotlib.use("TkAgg")

HYPERPARAMS = {
    "shill":
    {
        "DT": {
            "max_depth": 7,
            "splitter": "best",
            "criterion": "gini",
            "ccp_alpha": 0,
            "known_max_depth": 12,
            "known_max_min_leaf": 30,
            "n_estimators": 26,
            "boosted_max_depth": 7,
            "boosted_alpha": 0.02
        },
        "SVM": {
            "type": "rbf",
            "c": 1.,#.001,
            "max_iter": 100000,
            "gamma": 5,
            "coef0": 1,
            "degree": 3
        },
        "KNN": {
            "k": 3
        },
        "NN": {
            # "first_hidden_layer_node_count": 9,
            # "hidden_layers": [{"count": 100, "activation": "relu"}],
            # "loss_fn": 'sparse_categorical_crossentropy',
            "epochs": 200,
            "solver": "adam",
            "activation": "tanh",
            "hidden_layers": (50, 50),
            "learning_rate": 'adaptive',
            "alpha": .0005,  # 1e-5
        }
    },
    "cardio":
        {
            "DT": {
                "max_depth": 6,
                "splitter": "best",
                "criterion": "gini",
                "ccp_alpha": 0.001,
                "known_max_depth": 20,
                "known_max_min_leaf": 30,
                "n_estimators": 10,
                "boosted_max_depth": 3,
                "boosted_alpha": 0
            },
            "SVM": {
                "type": "poly",
                "c": 1.5,
                "max_iter": 100000,
                "gamma": 5,
                "coef0": 1,
                "degree": 3
            },
            "KNN": {
                "k": 3
            },
            "NN": {
                # "first_hidden_layer_node_count": 9,
                # "hidden_layers": [{"count": 100, "activation": "relu"}],
                # "loss_fn": 'sparse_categorical_crossentropy',
                "epochs": 200,
                "solver": "adam",
                "activation": "tanh",
                "hidden_layers": (50,50),
                "learning_rate": 'constant',
                "alpha": .5, #1e-5
            }
        },
}

METRICSSTORE = []

def get_datasets():
    files = ["./shill.csv", "./cardiotocogram.csv"]
    set_A = read_csv(files[0])
    set_B = read_csv(files[1])


    # Cleaning A
    # Drop first 3 columns, they aren't relevant to this study
    set_A = set_A.drop(['Record_ID', 'Auction_ID', 'Bidder_ID'], axis=1)
    set_A.dropna(inplace=True)

    set_B.dropna(inplace=True)
    return {
               "shill": {"features": set_A.drop(["Class"], axis=1), "class": set_A["Class"]},
               "cardio": {"features": set_B.drop(["Class"], axis=1), "class": set_B["Class"]-1},
           }, files

def get_hyperparams(dataset_name, algorithm):
    return HYPERPARAMS[dataset_name][algorithm]

def plot_roc(fpr, tpr, fn, label=None):
    # Draw lines
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')

    # Create plot
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate"), plt.ylabel("Recall")
    plt.tight_layout()
    plt.savefig("./%s" % fn)
    plt.close()

def plot_learning_curve2(estimator, fn, title, X, y, ylim=None, cv=None,
                    n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator, X, y, cv=cv, n_jobs=1, train_sizes=train_sizes, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation Score")

    plt.legend(loc="best")

    plt.savefig("./%s" % fn)
    plt.close()

    # timing
    time_mean = np.mean(fit_times, axis=1)

    # Draw lines
    plt.plot(train_sizes, time_mean, label="Fit Time")

    # Create plot
    plt.title("Scalability (w/ regards to time)")
    plt.xlabel("Training Set Size"), plt.ylabel("Time"), plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("./%s_scale" % fn)
    plt.close()

    return train_scores, train_sizes, test_scores, fit_times

def get_runtime_avgs(clf, features_test):
    times = []
    for i in range(0,10):
        t1 = time.time()
        ypred = clf.predict(features_test)
        times.append(round(time.time() - t1, 3))
    return np.mean(times)


def decision_tree_experiment(dataset, hparams, output_fn_base):
    logs = []
    metrics_dictionary = {}
    print("----Running Decision Tree Experiment-----")
    print("Hyperparameters Used: ")
    print(hparams)
    X = dataset["features"]
    y = dataset["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,test_size=.2, train_size=.8)

    # tree_clf = DecisionTreeClassifier()

    # # # grid search (comment out in production)
    # parameter_space = {
    #     'max_depth': list(range(hparams["known_max_depth"]+2)[1:]),
    #     "ccp_alpha": [.001,.1, .2,.3,.4,.5,.6,.7,.8,.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5],
    #     'criterion': ['gini', 'entropy']
    # }
    # baseclf = DecisionTreeClassifier()
    # clf = GridSearchCV(baseclf, parameter_space, n_jobs=1, cv=3)
    # clf.fit(X_train, y_train)
    # print('Best parameters found:\n', clf.best_params_)


    tree_clf = DecisionTreeClassifier(max_depth=hparams["max_depth"], splitter=hparams["splitter"], criterion=hparams["criterion"], ccp_alpha=hparams["ccp_alpha"])


    # learning curve
    cv = ShuffleSplit(n_splits=100, test_size=0.2)
    train_scores, train_sizes, validation_scores, fit_times = plot_learning_curve2(tree_clf, "%s/DT/Curve_%s" % (output_fn_base, output_fn_base), "Unboosted DT Learning Curve for %s" % output_fn_base, X, y, ylim=None, cv=cv, n_jobs=4)
    metrics_dictionary["train_scores"] = train_scores
    metrics_dictionary["train_sizes"] = train_sizes
    metrics_dictionary["validation_scores"] = validation_scores
    metrics_dictionary["fit_times"] = fit_times

    tree_clf.fit(X=X_train, y=y_train)
    runtimes = get_runtime_avgs(tree_clf, X_test)
    # confusion matrix
    matrix = create_confusion_matrix(tree_clf, X, y, 10, "%s/DT/Confusion_%s" % (output_fn_base, output_fn_base))

    # let's show the actual tree
    export_graphviz(tree_clf,out_file=("%s/DT/%s.dot" % (output_fn_base, output_fn_base)),feature_names=dataset["features"].columns, rounded=True,filled=True)
    (graph,) = pydot.graph_from_dot_file('%s/DT/%s.dot' % (output_fn_base, output_fn_base))
    graph.write_png('%s/DT/%s.png' % (output_fn_base, output_fn_base))

    # classification report:
    logs.append(classification_report(y_test, tree_clf.predict(X_test)))

    # cross validate:
    cvd = cross_validate(tree_clf, X.values, y.values, cv=10, scoring={'accuracy:': make_scorer(accuracy_score), 'precision': make_scorer(precision_score,average= 'micro'), 'recall': make_scorer(recall_score,average= 'micro'), 'f1_score':make_scorer(f1_score,average= 'micro')},)
    metrics_dictionary["cvd"] = cvd



    # now run experiments for the same metric but relative to hyperparameters
    # pruning is done with min_samples leaf, max_depth, ccp_alpha
    # info on ccp_alpha: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

    ##################################
    # CCP_ALPHA

    path = tree_clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    ccp_alphas= list(ccp_alphas)
    ccp_alphas.sort()
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]))

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1])
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.savefig("%s/DT/IMPURITYVALPHA_%s" % (output_fn_base, output_fn_base))
    plt.close()

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Alpha Impact on Score - DT For %s" % output_fn_base)
    ax.plot(ccp_alphas, train_scores, label="In-Sample")
    ax.plot(ccp_alphas, test_scores, label="Out-Of-Sample")
    ax.legend()
    plt.savefig("%s/DT/CCPALPHA_%s" % (output_fn_base, output_fn_base))
    plt.close()

    ##################################
    # Max_depth

    max_depths = list(range(hparams["known_max_depth"]+2)[1:])
    clfs = []
    for depth in max_depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    print("Number of nodes in the last tree is: {} with max_depth: {}".format(
        clfs[-1].tree_.node_count, max_depths[-1]))

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("max_depth")
    ax.set_ylabel("accuracy")
    ax.set_title("Max Depth Impact on Score - DT For %s"%output_fn_base)
    ax.plot(max_depths, train_scores, label="In-Sample")
    ax.plot(max_depths, test_scores, label="Out-Of-Sample")
    ax.legend()
    plt.savefig("%s/DT/DEPTH_%s" % (output_fn_base, output_fn_base))
    plt.close()

    ##################################
    # min_samples_leaf

    min_samples_leaf_params = list(range(1,hparams["known_max_min_leaf"], 1)[1:])
    clfs = []
    for leaf_size in min_samples_leaf_params:
        clf = DecisionTreeClassifier(min_samples_leaf=leaf_size)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    print("Number of nodes in the last tree is: {} with min_samples_leaf: {}".format(
        clfs[-1].tree_.node_count, min_samples_leaf_params[-1]))

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]


    fig, ax = plt.subplots()
    ax.set_xlabel("min_samples_leaf")
    ax.set_ylabel("accuracy")
    ax.set_title("Min Leaf Samples Impact on Score - DT For %s" % output_fn_base)
    ax.plot(min_samples_leaf_params, train_scores, label="In-Sample")
    ax.plot(min_samples_leaf_params, test_scores, label="Out-Of-Sample")
    ax.legend()
    plt.savefig("%s/DT/LEAF_%s" % (output_fn_base, output_fn_base))
    plt.close()

    logs.append("\tHyperparameters: \n")
    logs.append("\t%s" % str(hparams))
    logs.append("\n\n\tConfusion Matrix: \n")
    logs.append("\t%s\n" % str(matrix).replace("\n", "\n\t"))
    logs.append("\n\tMean Accuracy %.05f" % cvd["test_precision"].mean())
    logs.append("\n\tMean Precision Score of positive examples %.05f" % cvd["test_precision"].mean())
    logs.append("\n\tMean Recall Score of positive examples %.05f" % cvd["test_recall"].mean())
    logs.append("\n\tF1 Score of positive examples %.05f\n" % cvd["test_f1_score"].mean())
    logs.append("\n\tMean Query Time %.05f\n" % runtimes)

    return logs, metrics_dictionary


def boosted_decision_tree_experiment(dataset, hparams, output_fn_base):
    metrics_dictionary = {}
    logs = []
    print("----Running Boosted Decision Tree Experiment-----")
    print("Hyperparameters Used: ")
    print(hparams)
    X = dataset["features"]
    y = dataset["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,test_size=.2, train_size=.8)

    tree_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=hparams["boosted_max_depth"], splitter=hparams["splitter"], criterion=hparams["criterion"]),n_estimators=hparams["n_estimators"])


    # # # grid search (comment out in production)
    # parameter_space = {
    #     # 'max_depth': list(range(hparams["known_max_depth"]+2)[1:]),
    #     # "ccp_alpha": [.001,.1, .2,.3,.4,.5,.6,.7,.8,.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5],
    #     "n_estimators": range(10,100)
    # }
    # baseclf = AdaBoostClassifier(DecisionTreeClassifier())
    # clf = GridSearchCV(baseclf, parameter_space, n_jobs=1, cv=3)
    # clf.fit(X_train, y_train)
    # print('Best parameters found:\n', clf.best_params_)


    # learning curve
    cv = ShuffleSplit(n_splits=100, test_size=0.2)
    train_scores, train_sizes, validation_scores, fit_times = plot_learning_curve2(tree_clf, "%s/BoostedDT/Curve_%s" % (output_fn_base, output_fn_base), "Adaboost Learning Curve for %s" % output_fn_base, X, y, ylim=None, cv=cv, n_jobs=4)
    metrics_dictionary["train_scores"] = train_scores
    metrics_dictionary["train_sizes"] = train_sizes
    metrics_dictionary["validation_scores"] = validation_scores
    metrics_dictionary["fit_times"] = fit_times

    tree_clf.fit(X=X_train, y=y_train)
    runtimes = get_runtime_avgs(tree_clf, X_test)
    # confusion matrix
    matrix = create_confusion_matrix(tree_clf, X, y, 10, "%s/BoostedDT/Confusion_%s" % (output_fn_base, output_fn_base))

    # classification report:
    logs.append(classification_report(y_test, tree_clf.predict(X_test)))

    cvd = cross_validate(tree_clf, X.values, y.values, cv=10, scoring={'accuracy:': make_scorer(accuracy_score), 'precision': make_scorer(precision_score,average= 'micro'), 'recall': make_scorer(recall_score,average= 'micro'), 'f1_score':make_scorer(f1_score,average= 'micro')},)
    metrics_dictionary["cvd"] = cvd

    # # now run experiments for the same metric but relative to hyperparameters
    # # pruning is done with min_samples leaf, max_depth, ccp_alpha
    # # info on ccp_alpha: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

    ##################################
    # CCP_ALPHA

    path = tree_clf.base_estimator.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    ccp_alphas= list(ccp_alphas)
    ccp_alphas.sort()
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = AdaBoostClassifier(DecisionTreeClassifier(ccp_alpha=ccp_alpha))
        clf.fit(X_train, y_train)
        clfs.append(clf)


    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Alpha Impact on Score - Boosted DT For %s" % output_fn_base)
    ax.plot(ccp_alphas, train_scores, label="In-Sample")
    ax.plot(ccp_alphas, test_scores, label="Out-Of-Sample")
    ax.legend()
    plt.savefig("%s/BoostedDT/CCPALPHA_%s" % (output_fn_base, output_fn_base))
    plt.close()

    ##################################
    # Max_depth

    max_depths = list(range(hparams["known_max_depth"]+2)[1:])
    clfs = []
    for depth in max_depths:
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth))
        clf.fit(X_train, y_train)
        clfs.append(clf)

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("max_depth")
    ax.set_ylabel("accuracy")
    ax.set_title("Max Depth Impact on Score - Boosted DT For %s" % output_fn_base)
    ax.plot(max_depths, train_scores, label="In-Sample")
    ax.plot(max_depths, test_scores, label="Out-Of-Sample")
    ax.legend()
    plt.savefig("%s/BoostedDT/DEPTH_%s" % (output_fn_base, output_fn_base))
    plt.close()

    ##################################
    # n_estimators

    estimator_count = range(10,100)
    clfs = []
    for est in estimator_count:
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=hparams["max_depth"]),n_estimators=est)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("# of Estimators")
    ax.set_ylabel("accuracy")
    ax.set_title("Number of Estimators Impact on Score - Boosted DT For %s" % output_fn_base)
    ax.plot(estimator_count, train_scores, label="In-Sample")
    ax.plot(estimator_count, test_scores, label="Out-Of-Sample")
    ax.legend()
    plt.savefig("%s/BoostedDT/Estimators_%s" % (output_fn_base, output_fn_base))
    plt.close()

    logs.append("\tHyperparameters: \n")
    logs.append("\t%s" % str(hparams))
    logs.append("\n\n\tConfusion Matrix: \n")
    logs.append("\t%s\n" % str(matrix).replace("\n", "\n\t"))
    logs.append("\n\tMean Accuracy %.05f" % cvd["test_precision"].mean())
    logs.append("\n\tMean Precision Score of positive examples %.05f" % cvd["test_precision"].mean())
    logs.append("\n\tMean Recall Score of positive examples %.05f" % cvd["test_recall"].mean())
    logs.append("\n\tF1 Score of positive examples %.05f\n" % cvd["test_f1_score"].mean())
    logs.append("\n\tMean Query Time %.05f\n" % runtimes)
    return logs, metrics_dictionary

def svm_experiment(dataset, hparams, output_fn_base):
    logs = []
    metrics_dictionary = {}
    print("----Running SVM Experiment-----")
    print("Hyperparameters Used: ")
    print(hparams)
    X = dataset["features"]
    y = dataset["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,test_size=.2, train_size=.8)

    # # grid search (comment out in production)
    # parameter_space = {
    #     'kernel': ["rbf", "poly"],
    #     "C": [.001,.1, .2,.3,.4,.5,.6,.7,.8,.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
    # }
    # baseclf = SVC()
    # clf = GridSearchCV(baseclf, parameter_space, n_jobs=1, cv=3)
    # clf.fit(X_train, y_train)
    # print('Best parameters found:\n', clf.best_params_)


    # create basic SVM
    svm_clf = None
    if (hparams["type"] == "linear"):
        svm_clf = Pipeline([("scaler", StandardScaler()),("svm_clf", LinearSVC(C=hparams["c"], loss="hinge", max_iter=hparams["max_iter"]))])
    elif(hparams["type"] == "gaussian"):
        svm_clf = Pipeline([("scaler", StandardScaler()),("svm_clf", SVC(kernel="rbf", gamma=hparams["gamma"],C=hparams["c"], max_iter=hparams["max_iter"]))])
    else:
        svm_clf = Pipeline([("scaler", StandardScaler()),("svm_clf", SVC(kernel="poly", degree=hparams["degree"],C=hparams["c"], coef0=hparams["coef0"], max_iter=hparams["max_iter"]))])


    # learning curve
    cv = ShuffleSplit(n_splits=100, test_size=0.2)
    train_scores, train_sizes, validation_scores, fit_times = plot_learning_curve2(svm_clf, "%s/SVM/Curve_%s" % (output_fn_base, output_fn_base), "SVM Learning Curve for %s" % output_fn_base, X, y, ylim=None, cv=cv, n_jobs=4)
    metrics_dictionary["train_scores"] = train_scores
    metrics_dictionary["train_sizes"] = train_sizes
    metrics_dictionary["validation_scores"] = validation_scores
    metrics_dictionary["fit_times"] = fit_times


    svm_clf.fit(X=X_train, y=y_train)
    runtimes = get_runtime_avgs(svm_clf, X_test)
    # classification report:
    logs.append(classification_report(y_test, svm_clf.predict(X_test)))


    cvs = cross_val_score(svm_clf, X, y, cv=10, scoring='accuracy')
    print("Mean 10-fold CV Score: %.02f" % cvs.mean())

    # # precision and recall
    # is it binary?
    avg_precision_s = None
    precision_s = None
    recall_s = None
    f1_s = None
    if(dataset["class"].nunique() == 2):
        y_score = svm_clf.decision_function(X_test)
        avg_precision_s = average_precision_score(y_test, y_score)
        # define 'positive values'
        y_train_pos = (y_train == 1)
        y_train_pred = cross_val_predict(svm_clf, X_train, y_train_pos, cv=5)
        precision_s = precision_score(y_train_pos, y_train_pred)
        recall_s = recall_score(y_train_pos, y_train_pred)
        f1_s = f1_score(y_train_pos, y_train_pred)
        # plot roc curve
        y_scores = cross_val_predict(svm_clf, X_train, y_train_pos, cv=5, method="decision_function")
        fpr, tpr, thresh = roc_curve(y_train_pos, y_scores)
        plot_roc(fpr,tpr,"%s/SVM/ROC_%s" % (output_fn_base, output_fn_base))

    # decision function
    d_func = svm_clf.decision_function(X)

    # confusion matrix
    matrix = create_confusion_matrix(svm_clf, X, y, 10, "%s/SVM/Confusion_%s" % (output_fn_base, output_fn_base))

    # experiment with hyperparams
    #

    ##################################
    # C metric

    c_opts = [.001,.1, .2,.3,.4,.5,.6,.7,.8,.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]

    rbf_scores_test = []
    poly_scores_test = []
    rbf_scores_train = []
    poly_scores_train = []
    for c in c_opts:

        rbf_clf = Pipeline([("scaler", StandardScaler()),("rbf_clf", SVC(kernel="rbf", gamma=hparams["gamma"],C=c, max_iter=hparams["max_iter"]))])
        poly_svm_clf = Pipeline([("scaler", StandardScaler()),("svm_clf", SVC(kernel="poly", degree=hparams["degree"],C=c, coef0=hparams["coef0"], max_iter=hparams["max_iter"]))])
        rbf_clf.fit(X_train, y_train)
        poly_svm_clf.fit(X_train, y_train)
        rbf_scores_test.append(rbf_clf.score(X_test, y_test))
        poly_scores_test.append(poly_svm_clf.score(X_test, y_test))
        rbf_scores_train.append(rbf_clf.score(X_train, y_train))
        poly_scores_train.append(poly_svm_clf.score(X_train, y_train))

    fig, ax = plt.subplots()
    ax.set_xlabel("C value")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs C Value for training and testing SVM")
    ax.plot(c_opts, rbf_scores_test, label="RBF Kernel Test Scores")
    ax.plot(c_opts, rbf_scores_train, label="RBF Kernel Train Scores")
    ax.plot(c_opts, poly_scores_test, label="Poly Kernel Test Scores")
    ax.plot(c_opts, poly_scores_train, label="Poly Kernel Train Scores")
    ax.legend()
    plt.savefig("%s/SVM/RBFVSVM_C%s" % (output_fn_base, output_fn_base))
    plt.close()


    ##################################
    # Degree on poly metric

    degrees = [1,2,3,4,5,6,7,8,9,10,11,12]

    poly_scores_test = []
    poly_scores_train = []

    for degree in degrees:

        poly_svm_clf = Pipeline([("scaler", StandardScaler()),("svm_clf", SVC(kernel="poly", degree=degree,C=hparams["c"], coef0=hparams["coef0"], max_iter=hparams["max_iter"]))])
        poly_svm_clf.fit(X_train, y_train)
        poly_scores_test.append(poly_svm_clf.score(X_test, y_test))
        poly_scores_train.append(poly_svm_clf.score(X_train, y_train))

    fig, ax = plt.subplots()
    ax.set_xlabel("Degree")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs Degree Value for training and testing SVM")
    ax.plot(degrees, poly_scores_test, label="Poly Kernel Test Scores")
    ax.plot(degrees, poly_scores_train, label="Poly Kernel Train Scores")
    ax.legend()
    plt.savefig("%s/SVM/POLYSVM_DEGREE_%s" % (output_fn_base, output_fn_base))
    plt.close()

    logs.append("\tHyperparameters: \n")
    logs.append("\t%s" % str(hparams))
    logs.append("\n\n\tConfusion Matrix: \n")
    logs.append("\t%s\n" % str(matrix).replace("\n", "\n\t"))
    logs.append("\n\tMean 10-fold CV Score %.02f\n" % cvs.mean())
    logs.append("\n\tDecision Function on Test:\n\t%s \n" % str(d_func).replace("\n", "\n\t"))
    if(avg_precision_s is not None):
        logs.append("\n\tAVG Precision Score %.02f\n" % avg_precision_s)
    if(precision_s is not None):
        logs.append("\n\tPrecision Score of positive examples %.02f\n" % precision_s)
    if(recall_s is not None):
        logs.append("\n\tRecall Score of positive examples %.02f\n" % recall_s)
    if(f1_s is not None):
        logs.append("\n\tF1 Score of positive examples %.02f\n" % f1_s)
    logs.append("\n\tMean Query Time %.05f\n" % runtimes)
    return logs, metrics_dictionary

def knn_experiment(dataset, hparams, output_fn_base):
    logs = []
    metrics_dictionary = {}
    print("----Running KNN Experiment-----")
    print("Hyperparameters Used: ")
    print(hparams)
    X = dataset["features"]
    y = dataset["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,test_size=.2, train_size=.8)

    # # grid search (comment out in production)
    # parameter_space = {
    #     'n_neighbors': range(1,60),
    # }
    # baseclf = KNeighborsClassifier()
    # clf = GridSearchCV(baseclf, parameter_space, n_jobs=1, cv=3)
    # clf.fit(X_train, y_train)
    # print('Best parameters found:\n', clf.best_params_)


    neigh_clf = KNeighborsClassifier(n_neighbors=hparams["k"])

    # learning curve
    cv = ShuffleSplit(n_splits=100, test_size=0.2)
    train_scores, train_sizes, validation_scores, fit_times = plot_learning_curve2(neigh_clf, "%s/KNN/Curve_%s" % (output_fn_base, output_fn_base), "KNN Learning Curve for %s" % output_fn_base, X, y, ylim=None, cv=cv, n_jobs=4)
    metrics_dictionary["train_scores"] = train_scores
    metrics_dictionary["train_sizes"] = train_sizes
    metrics_dictionary["validation_scores"] = validation_scores
    metrics_dictionary["fit_times"] = fit_times


    neigh_clf.fit(X_train, y_train)
    runtimes = get_runtime_avgs(neigh_clf, X_test)
    # classification report:
    logs.append(classification_report(y_test, neigh_clf.predict(X_test)))

    # confusion matrix
    matrix = create_confusion_matrix(neigh_clf, X, y, 10, "%s/KNN/Confusion_%s" % (output_fn_base, output_fn_base))

    # cross validation score
    cvs = cross_val_score(neigh_clf, X, y, cv=10, scoring='accuracy')
    print("Mean 10-fold CV Score: %.02f" % cvs.mean())

    ##################################
    # K metric

    k_vals = range(1,60)

    k_scores_test = []
    k_scores_train = []
    for k in k_vals:
        test_neigh_clf = KNeighborsClassifier(n_neighbors=k)
        test_neigh_clf.fit(X_train, y_train)
        k_scores_test.append(test_neigh_clf.score(X_test, y_test))
        k_scores_train.append(test_neigh_clf.score(X_train, y_train))

    fig, ax = plt.subplots()
    ax.set_xlabel("k")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs k in kNN")
    ax.plot(k_vals, k_scores_test, label="Out-Of-Sample")
    ax.plot(k_vals, k_scores_train, label="In-Sample")

    ax.legend()
    plt.savefig("%s/KNN/knn_k_%s" % (output_fn_base, output_fn_base))
    plt.close()

    logs.append("\tHyperparameters: \n")
    logs.append("\t%s" % str(hparams))
    logs.append("\n\n\tConfusion Matrix: \n")
    logs.append("\t%s\n" % str(matrix).replace("\n", "\n\t"))
    logs.append("\n\tMean 10-fold CV Score %.02f\n" % cvs.mean())
    logs.append("\n\tMean Query Time %.05f\n" % runtimes)
    return logs, metrics_dictionary

# def neural_network_experiment(dataset, hparams, output_fn_base):
#     logs = []
#     print("----Running Neural Network Experiment-----")
#     print("Hyperparameters Used: ")
#     print(hparams)
#     X = dataset["features"]
#     y = dataset["class"]
#
#     nn_model = keras.models.Sequential()
#     nn_model.add(keras.layers.Dense(X.shape[1], activation='relu',))
#     for layer_info in hparams["hidden_layers"]:
#         nn_model.add(keras.layers.Dense(layer_info['count'], activation=layer_info['activation']))
#     nn_model.add(keras.layers.Dense(dataset["class"].nunique(), activation='sigmoid')) # output layer
#     nn_model.compile(loss=hparams["loss_fn"], optimizer='adam', metrics=['accuracy'])
#
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,test_size=.2, train_size=.8)
#     train_info = nn_model.fit(X_train, y_train, epochs=hparams["epochs"])
#
#     # classification report:
#     logs.append(classification_report(y_test, nn_model.predict(X_test)))
#
#     pd.DataFrame(train_info.history).plot()
#     plt.grid(True)
#     plt.xlabel("Epoch")
#     plt.title("Network Accuracy & Loss During Training")
#     plt.gca().set_ylim(0,1)
#     plt.savefig("%s/NN/nn_curve_%s" % (output_fn_base, output_fn_base))
#     plt.close()
#
#     loss_val = train_info.history['loss']
#     epochs = range(1, hparams["epochs"]+1)
#     plt.plot(epochs, loss_val, label='Loss')
#     plt.title('Network Loss Over Training')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.savefig("%s/NN/nn_LOSSONLY_%s" % (output_fn_base, output_fn_base))
#     plt.close()
#
#     accuracy_val = train_info.history['accuracy']
#     plt.plot(epochs, accuracy_val, label='Accuracy')
#     plt.title('Network Accuracy Over Training')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.savefig("%s/NN/nn_ACCONLY_%s" % (output_fn_base, output_fn_base))
#     plt.close()
#
#
#
#     # define 10-fold cross validation test harness
#     kfold = StratifiedKFold(n_splits=10, shuffle=True)
#     cvscores = []
#     print("Cross Validation Testing:")
#
#     for train, test in kfold.split(X, y):
#         # create model
#         nn_model = keras.models.Sequential()
#         nn_model.add(keras.layers.Dense(X.shape[1], activation='relu', ))
#         for layer_info in hparams["hidden_layers"]:
#             nn_model.add(keras.layers.Dense(layer_info['count'], activation=layer_info['activation']))
#         nn_model.add(keras.layers.Dense(dataset["class"].nunique(), activation='sigmoid'))  # output layer
#
#         nn_model.compile(loss=hparams["loss_fn"], optimizer='adam', metrics=['accuracy'])
#
#         # Fit the model
#         nn_model.fit(X.to_numpy()[train], y.to_numpy()[train], epochs=hparams["epochs"], verbose=0)
#         # evaluate the model
#         scores = nn_model.evaluate(X.to_numpy()[test], y.to_numpy()[test], verbose=0)
#         print("%s: %.2f%%" % (nn_model.metrics_names[1], scores[1] * 100))
#         cvscores.append(scores[1] * 100)
#     print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
#
#     logs.append("\tHyperparameters: \n")
#     logs.append("\t%s" % str(hparams))
#     logs.append("\n\tMean Accuracy from Cross Validation: %.2f%% (+/- %.2f%%)\n" % (np.mean(cvscores), np.std(cvscores)))
#     logs.append("\n\tCV Scores:\n\t%s\n" % str(cvscores))
#
#     return logs

def neural_network_experiment(dataset, hparams, output_fn_base):
    logs = []
    metrics_dictionary={}
    print("----Running Neural Network Experiment-----")
    print("Hyperparameters Used: ")
    print(hparams)
    X = dataset["features"]
    y = dataset["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,test_size=.2, train_size=.8)
    # scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # # grid search (comment out in production)
    # parameter_space = {
    #     'hidden_layer_sizes': [(50, 50,),(100,)],
    #     'activation': ['tanh', 'relu'],
    #     'solver': ['sgd', 'adam'],
    #     'alpha': [.0001,.0005,.001,.005,.01,.05,.1,.5],
    #     'learning_rate': ['constant', 'adaptive'],
    # }
    # mlp = MLPClassifier()
    # clf = GridSearchCV(mlp, parameter_space, n_jobs=1, cv=3)
    # clf.fit(X_train, y_train)
    # print('Best parameters found:\n', clf.best_params_)


    nn_clf = MLPClassifier(solver=hparams["solver"], alpha=hparams["alpha"],hidden_layer_sizes=hparams["hidden_layers"], max_iter=hparams["epochs"], activation=hparams["activation"], early_stopping=True, learning_rate=hparams["learning_rate"])

    # learning curve
    cv = ShuffleSplit(n_splits=100, test_size=0.2)
    train_scores, train_sizes, validation_scores, fit_times = plot_learning_curve2(nn_clf, "%s/NN/Curve_%s" % (output_fn_base, output_fn_base), "NN Learning Curve for %s" % output_fn_base, X, y, ylim=None, cv=cv, n_jobs=4)
    metrics_dictionary["train_scores"] = train_scores
    metrics_dictionary["train_sizes"] = train_sizes
    metrics_dictionary["validation_scores"] = validation_scores
    metrics_dictionary["fit_times"] = fit_times

    nn_clf.fit(X_train, y_train)
    runtimes = get_runtime_avgs(nn_clf, X_test)
    # classification report:
    logs.append(classification_report(y_test, nn_clf.predict(X_test)))

    # confusion matrix
    matrix = create_confusion_matrix(nn_clf, X, y, 10, "%s/NN/Confusion_%s" % (output_fn_base, output_fn_base))

    # cross validation score
    cvs = cross_val_score(nn_clf, X, y, cv=10, scoring='accuracy')
    print("Mean 10-fold CV Score: %.02f" % cvs.mean())

    ##################################
    # Alpha metric

    alphas = [.0001,.0005,.001,.005,.01,.05,.1,.5,.6,.7,.8,.9,1,1.2,1.4,1.6,1.8,1.9,2,2.2,2.4,2.6,2.8,3,4,5,6,7,8,9,10,11,12,13,14,15]

    scores_test = []
    scores_train = []
    for a in alphas:
        nn_clf = MLPClassifier(solver=hparams["solver"], alpha=a,
                               hidden_layer_sizes=hparams["hidden_layers"], max_iter=hparams["epochs"],
                               activation=hparams["activation"], early_stopping=True, learning_rate=hparams["learning_rate"])

        nn_clf.fit(X_train, y_train)

        scores_test.append(nn_clf.score(X_test, y_test))
        scores_train.append(nn_clf.score(X_train, y_train))

    fig, ax = plt.subplots()
    ax.set_xlabel("Alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Impact of alpha on Accuracy in NN")
    ax.plot(alphas, scores_test, label="Out-Of-Sample")
    ax.plot(alphas, scores_train, label="In-Sample")

    ax.legend()
    plt.savefig("%s/NN/nn_alpha_%s" % (output_fn_base, output_fn_base))
    plt.close()

    ##################################
    # Layer metric

    layers = [(50,),(50,50,),(50,50,50,),(50,50,50,50,),(50,50,50,50,50,),(50,50,50,50,50,50,),(50,50,50,50,50,50,50,)]

    scores_test = []
    scores_train = []
    for layer in layers:
        nn_clf = MLPClassifier(solver=hparams["solver"], alpha=hparams["alpha"],
                               hidden_layer_sizes=layer, max_iter=hparams["epochs"],
                               activation=hparams["activation"], early_stopping=True,
                               learning_rate=hparams["learning_rate"])

        nn_clf.fit(X_train, y_train)

        scores_test.append(nn_clf.score(X_test, y_test))
        scores_train.append(nn_clf.score(X_train, y_train))

    fig, ax = plt.subplots()
    ax.set_xlabel("Layer Count")
    ax.set_ylabel("accuracy")
    ax.set_title("Impact of layer count on accuracy in NN")
    ax.plot(range(1,8), scores_test, label="Out-Of-Sample")
    ax.plot(range(1,8), scores_train, label="In-Sample")

    ax.legend()
    plt.savefig("%s/NN/nn_layer_%s" % (output_fn_base, output_fn_base))
    plt.close()

    logs.append("\tHyperparameters: \n")
    logs.append("\t%s" % str(hparams))
    logs.append("\n\n\tConfusion Matrix: \n")
    logs.append("\t%s\n" % str(matrix).replace("\n", "\n\t"))
    logs.append("\n\tMean 10-fold CV Score %.02f\n" % cvs.mean())
    logs.append("\n\tMean Query Time %.05f\n" % runtimes)

    return logs, metrics_dictionary

def create_confusion_matrix(clf, X, y, cv, file_name):
    y_train_pred = cross_val_predict(clf, X, y, cv=cv)
    conf_mx = confusion_matrix(y,y_train_pred)
    plt.matshow(conf_mx,cmap=plt.cm.gray)
    plt.savefig(file_name)
    plt.close()
    return conf_mx

if __name__ == "__main__":
    print("#####################################################################")
    print("CS7641 ML - Supervised Learning Assignment Test Program")
    print("#####################################################################")
    datasets, filepaths = get_datasets()

    # Writing to file
    with open("metrics.txt", "w") as file1:
        file1.write("CS7641 ML - Supervised Learning Assignment Test Program \n")
        file1.write("Derek Chase Brown (dbrown381@gatech.edu) \n")
        file1.write("============ Datasets ============ \n")
        file1.writelines(filepaths)
        file1.write("\n============ Metrics ============ \n\n")
        # Test workbenches:
        for dataset_name in datasets:
            file1.write("DATASET: %s\n---------------------------------------------------------\n\n" % dataset_name)
            print("Testing on dataset: %s" % dataset_name )
            if not os.path.exists('%s/DT' % dataset_name):
                os.makedirs('%s/DT' % dataset_name)
            if not os.path.exists('%s/BoostedDT' % dataset_name):
                os.makedirs('%s/BoostedDT' % dataset_name)
            if not os.path.exists('%s/SVM' % dataset_name):
                os.makedirs('%s/SVM' % dataset_name)
            if not os.path.exists('%s/KNN' % dataset_name):
                os.makedirs('%s/KNN' % dataset_name)
            if not os.path.exists('%s/NN' % dataset_name):
                os.makedirs('%s/NN' % dataset_name)


            dt_logs, dt_metrics =decision_tree_experiment(datasets[dataset_name],get_hyperparams(dataset_name,"DT"), dataset_name)
            file1.write("DT Metrics ------------- \n")
            file1.writelines(dt_logs)
            file1.write("\n\tMean Train Time %.05f\n" % np.mean(dt_metrics["fit_times"], axis=1)[-1])

            boosted_logs, boosted_metrics = boosted_decision_tree_experiment(datasets[dataset_name], get_hyperparams(dataset_name, "DT"), dataset_name)
            file1.write("Boosted DT Metrics ------------- \n")
            file1.writelines(boosted_logs)
            file1.write("\n\tMean Train Time %.05f\n" % np.mean(boosted_metrics["fit_times"], axis=1)[-1])

            svm_logs, svm_metrics = svm_experiment(datasets[dataset_name], get_hyperparams(dataset_name, "SVM"), dataset_name)
            file1.write("SVM Metrics ------------- \n")
            file1.writelines(svm_logs)
            file1.write("\n\tMean Train Time %.05f\n" % np.mean(svm_metrics["fit_times"], axis=1)[-1])

            knn_logs, knn_metrics = knn_experiment(datasets[dataset_name], get_hyperparams(dataset_name, "KNN"), dataset_name)
            file1.write("KNN Metrics ------------- \n")
            file1.writelines(knn_logs)
            file1.write("\n\tMean Train Time %.05f\n" % np.mean(knn_metrics["fit_times"], axis=1)[-1])

            nn_logs, nn_metrics = neural_network_experiment(datasets[dataset_name], get_hyperparams(dataset_name, "NN"), dataset_name)
            file1.write("Neural Network Metrics ------------- \n")
            file1.writelines(nn_logs)
            file1.write("\n\tMean Train Time %.05f\n" % np.mean(nn_metrics["fit_times"], axis=1)[-1])

            print()

            plt.close()
            # Create plot - learning curves
            plt.plot(dt_metrics["train_sizes"], np.mean(dt_metrics["validation_scores"], axis=1),
                     label="Decision Tree (Unboosted)")
            plt.plot(boosted_metrics["train_sizes"], np.mean(boosted_metrics["validation_scores"], axis=1),
                     label="Decision Tree (Adaboost)")
            plt.plot(svm_metrics["train_sizes"], np.mean(svm_metrics["validation_scores"], axis=1),
                     label="SVM")
            plt.plot(knn_metrics["train_sizes"], np.mean(knn_metrics["validation_scores"], axis=1),
                     label="KNN")
            plt.plot(nn_metrics["train_sizes"], np.mean(nn_metrics["validation_scores"], axis=1),
                     label="NN")
            plt.title("Learning Curve Across All Models on %s" % dataset_name)
            plt.xlabel("Training Set Size"), plt.ylabel("Score"), plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig("./learning_curve_aggregated_%s" % dataset_name)
            plt.close()

            # Create plot - runtimes
            plt.plot(dt_metrics["train_sizes"], np.mean(dt_metrics["fit_times"], axis=1),
                     label="Decision Tree (Unboosted)")
            plt.plot(boosted_metrics["train_sizes"], np.mean(boosted_metrics["fit_times"], axis=1),
                     label="Decision Tree (Adaboost)")
            plt.plot(svm_metrics["train_sizes"], np.mean(svm_metrics["fit_times"], axis=1),
                     label="SVM")
            plt.plot(knn_metrics["train_sizes"], np.mean(knn_metrics["fit_times"], axis=1),
                     label="KNN")
            plt.plot(knn_metrics["train_sizes"], np.mean(knn_metrics["fit_times"], axis=1),
                     label="NN")
            plt.title("Scalability (w/ regards to time) on %s" % dataset_name)
            plt.xlabel("Training Set Size"), plt.ylabel("Time"), plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig("./runtime_curve_aggregated_%s" % dataset_name)
            plt.close()
