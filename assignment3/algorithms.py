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
    "cardio":
        {
            "NN": {
                # "first_hidden_layer_node_count": 9,
                # "hidden_layers": [{"count": 100, "activation": "relu"}],
                # "loss_fn": 'sparse_categorical_crossentropy',
                "epochs": 300,
                "solver": "adam",
                "activation": "tanh",
                "hidden_layers": (50,50),
                "learning_rate": 'constant',
                "alpha": .5, #1e-5
                "learning_rate_init": .4
            }
        },
    "cardio_pca":
        {
            "NN": {
                # "first_hidden_layer_node_count": 9,
                # "hidden_layers": [{"count": 100, "activation": "relu"}],
                # "loss_fn": 'sparse_categorical_crossentropy',
                "epochs": 300,
                "solver": "adam",
                "activation": "tanh",
                "hidden_layers": (50,50),
                "learning_rate": 'adaptive',
                "alpha": .0001, #1e-5
                "learning_rate_init": .4
            }
        },
    "cardio_ica":
        {
            "NN": {
                # "first_hidden_layer_node_count": 9,
                # "hidden_layers": [{"count": 100, "activation": "relu"}],
                # "loss_fn": 'sparse_categorical_crossentropy',
                "epochs": 300,
                "solver": "adam",
                "activation": "tanh",
                "hidden_layers": (50,50),
                "learning_rate": 'adaptive',
                "alpha": .001, #1e-5
                "learning_rate_init": .4
            }
        },
    "cardio_rp":
        {
            "NN": {
                # "first_hidden_layer_node_count": 9,
                # "hidden_layers": [{"count": 100, "activation": "relu"}],
                # "loss_fn": 'sparse_categorical_crossentropy',
                "epochs": 300,
                "solver": "adam",
                "activation": "relu",
                "hidden_layers": (50,50),
                "learning_rate": 'adaptive',
                "alpha": .1, #1e-5
                "learning_rate_init": .4
            }
        },
    "cardio_fa":
        {
            "NN": {
                # "first_hidden_layer_node_count": 9,
                # "hidden_layers": [{"count": 100, "activation": "relu"}],
                # "loss_fn": 'sparse_categorical_crossentropy',
                "epochs": 300,
                "solver": "adam",
                "activation": "relu",
                "hidden_layers": (50, 50),
                "learning_rate": 'adaptive',
                "alpha": .0005,  # 1e-5
                "learning_rate_init": .4
            }
        },
    "cardio_expectation_maximization_pca":
        {
            "NN": {
                # "first_hidden_layer_node_count": 9,
                # "hidden_layers": [{"count": 100, "activation": "relu"}],
                # "loss_fn": 'sparse_categorical_crossentropy',
                "epochs": 300,
                "solver": "sgd",
                "activation": "tanh",
                "hidden_layers": (50, 50),
                "learning_rate": 'adaptive',
                "alpha": .0001,  # 1e-5
                "learning_rate_init": .4
            }
        },
    "cardio_expectation_maximization_ica":
        {
            "NN": {
                # "first_hidden_layer_node_count": 9,
                # "hidden_layers": [{"count": 100, "activation": "relu"}],
                # "loss_fn": 'sparse_categorical_crossentropy',
                "epochs": 300,
                "solver": "sgd",
                "activation": "tanh",
                "hidden_layers": (50, 50),
                "learning_rate": 'adaptive',
                "alpha": .0001,  # 1e-5
                "learning_rate_init": .4
            }
        },
    "cardio_expectation_maximization_rp":
        {
            "NN": {
                # "first_hidden_layer_node_count": 9,
                # "hidden_layers": [{"count": 100, "activation": "relu"}],
                # "loss_fn": 'sparse_categorical_crossentropy',
                "epochs": 300,
                "solver": "sgd",
                "activation": "tanh",
                "hidden_layers": (50, 50),
                "learning_rate": 'adaptive',
                "alpha": .0001,  # 1e-5
                "learning_rate_init": .4
            }
        },
    "cardio_expectation_maximization_fa":
        {
            "NN": {
                # "first_hidden_layer_node_count": 9,
                # "hidden_layers": [{"count": 100, "activation": "relu"}],
                # "loss_fn": 'sparse_categorical_crossentropy',
                "epochs": 300,
                "solver": "sgd",
                "activation": "tanh",
                "hidden_layers": (50, 50),
                "learning_rate": 'adaptive',
                "alpha": .0001,  # 1e-5
                "learning_rate_init": .4
            }
        },
    "cardio_kmeans_pca":
        {
            "NN": {
                # "first_hidden_layer_node_count": 9,
                # "hidden_layers": [{"count": 100, "activation": "relu"}],
                # "loss_fn": 'sparse_categorical_crossentropy',
                "epochs": 300,
                "solver": "sgd",
                "activation": "tanh",
                "hidden_layers": (50, 50),
                "learning_rate": 'adaptive',
                "alpha": .0001,  # 1e-5
                "learning_rate_init": .4
            }
        },
    "cardio_kmeans_ica":
        {
            "NN": {
                # "first_hidden_layer_node_count": 9,
                # "hidden_layers": [{"count": 100, "activation": "relu"}],
                # "loss_fn": 'sparse_categorical_crossentropy',
                "epochs": 300,
                "solver": "sgd",
                "activation": "tanh",
                "hidden_layers": (50, 50),
                "learning_rate": 'adaptive',
                "alpha": .0001,  # 1e-5
                "learning_rate_init": .4
            }
        },
    "cardio_kmeans_rp":
        {
            "NN": {
                # "first_hidden_layer_node_count": 9,
                # "hidden_layers": [{"count": 100, "activation": "relu"}],
                # "loss_fn": 'sparse_categorical_crossentropy',
                "epochs": 300,
                "solver": "sgd",
                "activation": "tanh",
                "hidden_layers": (50, 50),
                "learning_rate": 'adaptive',
                "alpha": .0001,  # 1e-5
                "learning_rate_init": .4
            }
        },
    "cardio_kmeans_fa":
        {
            "NN": {
                # "first_hidden_layer_node_count": 9,
                # "hidden_layers": [{"count": 100, "activation": "relu"}],
                # "loss_fn": 'sparse_categorical_crossentropy',
                "epochs": 300,
                "solver": "adam",
                "activation": "tanh",
                "hidden_layers": (50, 50),
                "learning_rate": 'adaptive',
                "alpha": .0001,  # 1e-5
                "learning_rate_init": .4
            }
        },
}

METRICSSTORE = []

def get_datasets():
    files = [
             "./cardiotocogram.csv", "./cardio/PCADatasets/pca_cardio_17.csv", "./cardio/ICADatasets/ica_cardio_10.csv",
             "./cardio/RPDatasets/rp_cardio_7.csv", "./cardio/FADatasets/fa_cardio_8.csv",
             "./cardio/KMEANS/PCA/KMEANScardio_5.csv","./cardio/EM/PCA/EMcardio_4.csv",
             "./cardio/KMEANS/ICA/KMEANScardio_5.csv","./cardio/EM/ICA/EMcardio_9.csv",
             "./cardio/KMEANS/RP/KMEANScardio_4.csv","./cardio/EM/RP/EMcardio_5.csv",
             "./cardio/KMEANS/FA/KMEANScardio_5.csv","./cardio/EM/FA/EMcardio_5.csv"
    ]
    set_A = read_csv(files[0])
    set_B = read_csv(files[1])


    # Cleaning A
    # Drop first 3 columns, they aren't relevant to this study
    # set_A = set_A.drop(['Record_ID', 'Auction_ID', 'Bidder_ID'], axis=1)
    # set_A.dropna(inplace=True)

    set_B.dropna(inplace=True)
    return {
                "cardio": {"features": read_csv(files[0]).drop(["Class"], axis=1), "class": read_csv(files[0])["Class"] - 1},
                "cardio_pca": {"features": read_csv(files[1]).drop(["dfclass"], axis=1), "class": read_csv(files[1])["dfclass"]-1},
                "cardio_ica": {"features": read_csv(files[2]).drop(["dfclass"], axis=1), "class": read_csv(files[2])["dfclass"] - 1},
                "cardio_rp": {"features": read_csv(files[3]).drop(["dfclass"], axis=1), "class": read_csv(files[3])["dfclass"] - 1},
                "cardio_fa": {"features": read_csv(files[4]).drop(["dfclass"], axis=1), "class": read_csv(files[4])["dfclass"] - 1},

                "cardio_expectation_maximization_pca": {"features": read_csv(files[6]).drop(["dfclass"], axis=1), "class": read_csv(files[6])["dfclass"] - 1},
                "cardio_kmeans_pca": {"features": read_csv(files[5]).drop(["dfclass"], axis=1), "class": read_csv(files[5])["dfclass"] - 1},
               "cardio_expectation_maximization_ica": {"features": read_csv(files[8]).drop(["dfclass"], axis=1),
                                                       "class": read_csv(files[8])["dfclass"] - 1},
               "cardio_kmeans_ica": {"features": read_csv(files[7]).drop(["dfclass"], axis=1),
                                     "class": read_csv(files[7])["dfclass"] - 1},
               "cardio_expectation_maximization_rp": {"features": read_csv(files[10]).drop(["dfclass"], axis=1),
                                                       "class": read_csv(files[10])["dfclass"] - 1},
               "cardio_kmeans_rp": {"features": read_csv(files[9]).drop(["dfclass"], axis=1),
                                     "class": read_csv(files[9])["dfclass"] - 1},
               "cardio_expectation_maximization_fa": {"features": read_csv(files[12]).drop(["dfclass"], axis=1),
                                                       "class": read_csv(files[12])["dfclass"] - 1},
               "cardio_kmeans_fa": {"features": read_csv(files[11]).drop(["dfclass"], axis=1),
                                     "class": read_csv(files[11])["dfclass"] - 1},
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


    nn_clf = MLPClassifier(solver=hparams["solver"], alpha=hparams["alpha"],hidden_layer_sizes=hparams["hidden_layers"], max_iter=hparams["epochs"], activation=hparams["activation"], early_stopping=True, learning_rate=hparams["learning_rate"], learning_rate_init= hparams["learning_rate_init"])

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
                               activation=hparams["activation"], early_stopping=True, learning_rate=hparams["learning_rate"] , learning_rate_init= hparams["learning_rate_init"])

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
                               learning_rate=hparams["learning_rate"], learning_rate_init= hparams["learning_rate_init"])

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




    ##################################
    # learning rate metric

    learning_rates = [.0001,.0005,.001,.005,.01,.05,.1,.5,.6,.7,.8,.9]

    scores_test = []
    scores_train = []
    for rate in learning_rates:
        nn_clf = MLPClassifier(solver=hparams["solver"], alpha=hparams["alpha"],
                               hidden_layer_sizes=hparams["hidden_layers"], max_iter=hparams["epochs"],
                               activation=hparams["activation"], early_stopping=True, learning_rate=hparams["learning_rate"], learning_rate_init= rate)

        nn_clf.fit(X_train, y_train)

        scores_test.append(nn_clf.score(X_test, y_test))
        scores_train.append(nn_clf.score(X_train, y_train))

    fig, ax = plt.subplots()
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("accuracy")
    ax.set_title("Impact of changing learning rate - NN")
    ax.plot(learning_rates, scores_test, label="Out-Of-Sample")
    ax.plot(learning_rates, scores_train, label="In-Sample")

    ax.legend()
    plt.savefig("%s/NN/nn_learningrate_%s" % (output_fn_base, output_fn_base))
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
    metrics_dict = {}
    # Writing to file
    with open("nn_metrics.txt", "w") as file1:
        file1.write("CS7641 ML - Supervised Learning Assignment Test Program \n")
        file1.write("Derek Chase Brown (dbrown381@gatech.edu) \n")
        file1.write("============ Datasets ============ \n")
        file1.writelines(filepaths)
        file1.write("\n============ Metrics ============ \n\n")
        # Test workbenches:
        for dataset_name in datasets:
            file1.write("DATASET: %s\n---------------------------------------------------------\n\n" % dataset_name)
            print("Testing on dataset: %s" % dataset_name )
            if not os.path.exists('%s/NN' % dataset_name):
                os.makedirs('%s/NN' % dataset_name)

            nn_logs, nn_metrics = neural_network_experiment(datasets[dataset_name], get_hyperparams(dataset_name, "NN"), dataset_name)
            metrics_dict[dataset_name] = nn_metrics
            file1.write("Neural Network Metrics ------------- \n")
            file1.writelines(nn_logs)
            file1.write("\n\tMean Train Time %.05f\n" % np.mean(nn_metrics["fit_times"], axis=1)[-1])

            print()

            plt.close()


        dim_datasets = ["cardio","cardio_pca","cardio_ica","cardio_rp","cardio_fa"]
        cluster_datasets = ["cardio","cardio_expectation_maximization_pca", "cardio_kmeans_pca",
                            "cardio_expectation_maximization_ica","cardio_kmeans_ica",
                            "cardio_expectation_maximization_rp","cardio_kmeans_rp",
                            "cardio_expectation_maximization_fa","cardio_kmeans_fa"]






        #dim datasets
        for dataset_name in dim_datasets:
            plt.plot(metrics_dict[dataset_name]["train_sizes"], np.mean(metrics_dict[dataset_name]["validation_scores"], axis=1),
                 label=dataset_name)
        plt.title("Learning Curve Across All Dim Reduction Algorithms")
        plt.xlabel("Training Set Size"), plt.ylabel("Score"), plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig("./learning_curve_aggregated_allnn_dim")
        plt.close()

        for dataset_name in dim_datasets:
            plt.plot(metrics_dict[dataset_name]["train_sizes"], np.mean(metrics_dict[dataset_name]["fit_times"], axis=1),
                 label=dataset_name)
        plt.title("Scalability (w/ regards to time) - All Dim Reduction Algorithms")
        plt.xlabel("Training Set Size"), plt.ylabel("Time"), plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig("./runtime_curve_aggregated_all_dim")
        plt.close()

        # cluster datasets
        for dataset_name in cluster_datasets:
            plt.plot(metrics_dict[dataset_name]["train_sizes"], np.mean(metrics_dict[dataset_name]["validation_scores"], axis=1),
                 label=dataset_name)
        plt.title("Learning Curve Across All Dim Reduction/Cluster Algorithms")
        plt.xlabel("Training Set Size"), plt.ylabel("Score"), plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig("./learning_curve_aggregated_allnn_cluster")
        plt.close()

        for dataset_name in cluster_datasets:
            plt.plot(metrics_dict[dataset_name]["train_sizes"], np.mean(metrics_dict[dataset_name]["fit_times"], axis=1),
                 label=dataset_name)
        plt.title("Scalability (w/ regards to time) - All Dim Reduction/Cluster Algorithms")
        plt.xlabel("Training Set Size"), plt.ylabel("Time"), plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig("./runtime_curve_aggregated_all_cluster")
        plt.close()

