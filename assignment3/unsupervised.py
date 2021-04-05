from pandas import read_csv
import pandas as pd
from sklearn.model_selection import learning_curve, cross_val_score, train_test_split, cross_val_predict, StratifiedKFold, cross_validate, GridSearchCV, ShuffleSplit
import pydot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from pylab import MaxNLocator
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import SilhouetteVisualizer
import time
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection, johnson_lindenstrauss_min_dim
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import factor_analyzer as fana
# the following line might need to be commented out on some machines
matplotlib.use("TkAgg")
from sklearn.preprocessing import OneHotEncoder

HYPERPARAMS = {
    "shill":
    {
        "KMeans" : {
            "max_clusters": 20,
            "best": 5
        },
        "EM":
        {
            "max_components" : 20,
            "best": 8
        }

    },
    "cardio":
    {
        "KMeans" : {
            "max_clusters": 20,
            "best": 7
        },
        "EM":
        {
            "max_components": 20,
            "best": 5
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
               "shill": {"all": set_A, "features": set_A.drop(["Class"], axis=1), "class": set_A["Class"]},
               "cardio": {"all": set_B, "features": set_B.drop(["Class"], axis=1), "class": set_B["Class"]-1},
           }, files


def get_hyperparams(dataset_name, algorithm):
    return HYPERPARAMS[dataset_name][algorithm]

colors = ["red", "green", "blue", "black", "orange"]
color_idx = 0

def lineplot(x,y, filename, title, xlab,ylab, label=None):
    global colors
    global color_idx
    # Draw lines
    plt.plot(x,y, linewidth=2, label=label, color=colors[color_idx % len(colors)])
    color_idx += 1

    # Create plot
    plt.title(title)
    plt.xlabel(xlab), plt.ylabel(ylab)
    #plt.tight_layout()
    plt.grid()
    plt.xticks(range(x[0], x[-1]+1,2))

    plt.savefig("./%s" % filename)
    plt.close()
#https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam
def silhouetteplotkmeans(clusters_to_try, data, filename):
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    for j, i in enumerate(clusters_to_try):
        #km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100)
        km = clusters_to_try[j]
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[0 if j < 2 else 1][(j+2)%2],)
        visualizer.fit(data)
        visualizer.show(outpath="./%s.png" % filename)
    #plt.savefig("./%s" % filename)
    plt.close()

#https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam
def silhouetteplotgm(clusters_to_try, data, filename):
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    for j, i in enumerate(clusters_to_try):
        gm = GaussianMixture(n_components=i, n_init=10)
        visualizer = SilhouetteVisualizer(gm, colors='yellowbrick', ax=ax[0 if j < 2 else 1][(j + 2) % 2])
        visualizer.fit(data)
        visualizer.show(outpath="./%s.png" % filename)
    # plt.savefig("./%s" % filename)
    plt.close()


def kmeans_experiment(dataset, hparams, output_fn_base):
    logs = []
    metrics_dictionary = {}
    print("----Running k-Means Experiment-----")
    print("Hyperparameters Used: ")
    print(hparams)
    X = dataset["all"]

    silhouette_scores = []
    # https://scikit-learn.org/stable/modules/clustering.html
    # Ground truth labels not knwon - sillhouette scores are useful, the higher the score, the more defined clusters
    inertias = []
    fit_times = []
    ch_scores = []
    db_scores = []
    cluster_labels_by_num_component = []
    clusters_for_silhouette = []
    for k in range(2,hparams["max_clusters"]):
        kmeans = KMeans(n_clusters=k)
        t1 = time.time()
        kmeans.fit(X)
        fit_times.append(time.time() - t1)
        labels = kmeans.labels_
        ch_scores.append(calinski_harabasz_score(X,kmeans.labels_))
        silhouette_scores.append(silhouette_score(X,labels))
        inertias.append(kmeans.inertia_)
        db_scores.append(davies_bouldin_score(X,labels))
        cluster_labels_by_num_component.append(labels)# used to handle the following:
        # When you reproduced your clustering experiments on the datasets projected onto the new spaces created by ICA,
        # PCA, and RP, did you get the same clusters as before? Different clusters? Why? Why not?
        if(output_fn_base == "shill" and k == 5):
            lab = pd.DataFrame(kmeans.labels_).assign(dfclass=dataset["class"]).to_numpy()
            counts = [[],[]]
            for i in range(0,k):
                lab_temp = lab[lab[:, 0] == i]
                counts[0].append(len(lab_temp[lab_temp[:,1] == 0]))
                counts[1].append(len(lab_temp[lab_temp[:,1] == 1]))
            pd.DataFrame(np.array(counts)).to_csv("counts_shill_kmeans.csv")

        if(output_fn_base == "cardio" and k == 7):
            lab = pd.DataFrame(kmeans.labels_).assign(dfclass=dataset["class"]).to_numpy()
            counts = [[],[],[],[],[],[],[],[],[],[]]
            for i in range(0,k):
                lab_temp = lab[lab[:, 0] == i]
                counts[0].append(len(lab_temp[lab_temp[:,1] == 0]))
                counts[1].append(len(lab_temp[lab_temp[:,1] == 1]))
                counts[2].append(len(lab_temp[lab_temp[:, 1] == 2]))
                counts[3].append(len(lab_temp[lab_temp[:, 1] == 3]))
                counts[4].append(len(lab_temp[lab_temp[:, 1] == 4]))
                counts[5].append(len(lab_temp[lab_temp[:, 1] == 5]))
                counts[6].append(len(lab_temp[lab_temp[:, 1] == 6]))
                counts[7].append(len(lab_temp[lab_temp[:, 1] == 7]))
                counts[8].append(len(lab_temp[lab_temp[:, 1] == 8]))
                counts[9].append(len(lab_temp[lab_temp[:, 1] == 9]))
            pd.DataFrame(np.array(counts)).to_csv("counts_cardio_kmeans.csv")
        if k >= 4 and k <=7:
            clusters_for_silhouette.append(kmeans)

    metrics_dictionary["kmeans_fit_times"] = fit_times
    metrics_dictionary["kmeans_silhouette"] = silhouette_scores
    metrics_dictionary["kmeans_db"] = db_scores
    metrics_dictionary["cluster_x_tics"] = list(range(2,hparams["max_clusters"]))
    metrics_dictionary["inertia"] = inertias
    #silhouetteplotkmeans(clusters_for_silhouette, X, "%s/KMEANS/sil_%s" % (output_fn_base, output_fn_base))
    lineplot(x=list(range(2, hparams["max_clusters"])), y=inertias,
             filename="%s/KMEANS/inertia_%s" % (output_fn_base, output_fn_base),
             title="Inertia as K Increases - %s" % output_fn_base, xlab="K", ylab="Inertia", label="Inertia")

    lineplot(x=list(range(2, hparams["max_clusters"])), y=fit_times,
             filename="%s/KMEANS/fit_times_%s" % (output_fn_base, output_fn_base),
             title="Fit Runtime as K Increases - %s" % output_fn_base, xlab="K", ylab="Runtime", label="Runtime")

    lineplot(x=list(range(2, hparams["max_clusters"])), y=ch_scores,
             filename="%s/KMEANS/ch_scores_%s" % (output_fn_base, output_fn_base),
             title="Calinski-Harabasz Index as K Increases - %s" % output_fn_base, xlab="K", ylab="CH Index", label="CH Index")

    lineplot(x=list(range(2, hparams["max_clusters"])), y=db_scores,
             filename="%s/KMEANS/db_scores_%s" % (output_fn_base, output_fn_base),
             title="Davies-Bouldin Index as K Increases - %s" % output_fn_base, xlab="K", ylab="DB Index", label="DB Index")

    lineplot(x=list(range(2, hparams["max_clusters"])), y=silhouette_scores,
             filename="%s/KMEANS/silhouette_scores_%s" % (output_fn_base, output_fn_base),
             title="Silhouette Scores as K Increases - %s" % output_fn_base, xlab="K", ylab="Silhouette Score", label="Silhouette Score")

    # experiments with hard-coded best k:
    if not os.path.exists("%s/KMEANS/TSNE/" % output_fn_base):
        os.makedirs("%s/KMEANS/TSNE/" % output_fn_base)
    kmeans = KMeans(n_clusters=hparams["best"])
    kmeans.fit_predict(X)
    create_tsne(X,kmeans.labels_,"%s/KMEANS/TSNE/tsne_%s_kmeans" % (output_fn_base, output_fn_base))


    """
    {
        type: pca/ica/proj/custom
        kwargs: {..args}
    }
    
    """


    logs.append("\tHyperparameters: \n")
    logs.append("\t%s" % str(hparams))
    metrics_dictionary["cluster_labels_by_k"] = cluster_labels_by_num_component

    return logs, metrics_dictionary

def em_experiment(dataset, hparams, output_fn_base):
    logs = []
    metrics_dictionary = {}
    print("----Running Expectation Maximization Experiment-----")
    print("Hyperparameters Used: ")
    print(hparams)
    X = dataset["all"]

    silhouette_scores = []
    fit_times = []
    bic_scores = []
    aic_scores = []
    db_scores = []
    ch_scores = []
    cluster_labels_by_num_component = []
    for component_num in range(2,hparams["max_components"]):
        gm = GaussianMixture(n_components=component_num, n_init=10)
        t1 = time.time()
        gm.fit(X)
        fit_times.append(time.time() - t1)
        predicted = gm.predict(X)
        ch_scores.append(calinski_harabasz_score(X, predicted))
        silhouette_scores.append(silhouette_score(X, predicted))
        bic_scores.append(gm.bic(X))
        aic_scores.append(gm.aic(X))
        db_scores.append(davies_bouldin_score(X, predicted))
        cluster_labels_by_num_component.append(predicted) # used to handle the following:
        # When you reproduced your clustering experiments on the datasets projected onto the new spaces created by ICA,
        # PCA, and RP, did you get the same clusters as before? Different clusters? Why? Why not?
        if(output_fn_base == "shill" and component_num == 8):
            #pd.DataFrame(predicted).to_csv("./shill_em.csv",index=False)
            lab = pd.DataFrame(predicted).assign(dfclass=dataset["class"]).to_numpy()
            counts = [[],[]]
            for i in range(0,component_num):
                lab_temp = lab[lab[:, 0] == i]
                counts[0].append(len(lab_temp[lab_temp[:,1] == 0]))
                counts[1].append(len(lab_temp[lab_temp[:,1] == 1]))
            pd.DataFrame(np.array(counts)).to_csv("counts_shill_em.csv")

        if(output_fn_base == "cardio" and component_num == 5):
            # pd.DataFrame(predicted).to_csv("./cardio_em.csv",index=False)
            lab = pd.DataFrame(predicted).assign(dfclass=dataset["class"]).to_numpy()
            counts = [[], [], [], [], [], [], [], [], [], []]
            for i in range(0, component_num):
                lab_temp = lab[lab[:, 0] == i]
                counts[0].append(len(lab_temp[lab_temp[:, 1] == 0]))
                counts[1].append(len(lab_temp[lab_temp[:, 1] == 1]))
                counts[2].append(len(lab_temp[lab_temp[:, 1] == 2]))
                counts[3].append(len(lab_temp[lab_temp[:, 1] == 3]))
                counts[4].append(len(lab_temp[lab_temp[:, 1] == 4]))
                counts[5].append(len(lab_temp[lab_temp[:, 1] == 5]))
                counts[6].append(len(lab_temp[lab_temp[:, 1] == 6]))
                counts[7].append(len(lab_temp[lab_temp[:, 1] == 7]))
                counts[8].append(len(lab_temp[lab_temp[:, 1] == 8]))
                counts[9].append(len(lab_temp[lab_temp[:, 1] == 9]))
            pd.DataFrame(np.array(counts)).to_csv("counts_cardio_em.csv")
    metrics_dictionary["em_fit_times"] = fit_times
    metrics_dictionary["em_silhouette"] = silhouette_scores
    metrics_dictionary["em_db"] = db_scores
    #silhouetteplotgm([3,4,5,6], X, "%s/EM/sil_%s" % (output_fn_base, output_fn_base))
    lineplot(x=list(range(2, hparams["max_components"])), y=bic_scores,
             filename="%s/EM/bic_%s" % (output_fn_base, output_fn_base),
             title="Bayesian Information Criterior as Component Count Increases - %s" % output_fn_base, xlab="Num Components", ylab="BIC", label="BIC")

    lineplot(x=list(range(2, hparams["max_components"])), y=fit_times,
             filename="%s/EM/fit_times_%s" % (output_fn_base, output_fn_base),
             title="Fit Runtime as Component Count Increases - %s" % output_fn_base, xlab="Num Components", ylab="Runtime", label="Runtime")

    lineplot(x=list(range(2, hparams["max_components"])), y=aic_scores,
             filename="%s/EM/aic_scores_%s" % (output_fn_base, output_fn_base),
             title="Akaike Information Criterion as Component Count Increases - %s" % output_fn_base, xlab="Num Components", ylab="AIC", label="AIC")

    lineplot(x=list(range(2, hparams["max_components"])), y=silhouette_scores,
             filename="%s/EM/silhouette_scores_%s" % (output_fn_base, output_fn_base),
             title="Silhouette Scores as Component Count Increases - %s" % output_fn_base, xlab="Num Components", ylab="Silhouette Score", label="Silhouette Score")

    lineplot(x=list(range(2, hparams["max_components"])), y=db_scores,
             filename="%s/EM/db_scores_%s" % (output_fn_base, output_fn_base),
             title="Davies-Bouldin Index as Component Count Increases - %s" % output_fn_base, xlab="Num Components", ylab="DB Index", label="DB Index")

    lineplot(x=list(range(2, hparams["max_components"])), y=db_scores,
             filename="%s/EM/ch_scores_%s" % (output_fn_base, output_fn_base),
             title="Calinski Harabasz Score as Component Count Increases - %s" % output_fn_base, xlab="Num Components", ylab="CH Score", label="CH Score")

    # experiments with hard-coded best k:
    if not os.path.exists("%s/EM/TSNE/" % output_fn_base):
        os.makedirs("%s/EM/TSNE/" % output_fn_base)
    em = GaussianMixture(n_components=hparams["best"], n_init=10)
    labels = em.fit_predict(X)
    create_tsne(X,labels,"%s/EM/TSNE/tsne_%s_EM" % (output_fn_base, output_fn_base))


    logs.append("\tHyperparameters: \n")
    logs.append("\t%s" % str(hparams))
    metrics_dictionary["cluster_labels_by_k"] = cluster_labels_by_num_component

    return logs, metrics_dictionary


def pca_experiment(dataset, output_fn_base, perform_clustering=False, best_component_count=2):
    metrics_dictionary = {}
    max_k_clusters = 12
    max_n_components = 12
    pca_dim_max = dataset["all"].shape[1]-1
    # Reducing the dimensions of the data
    if not os.path.exists('%s/EM/PCA' % output_fn_base):
        os.makedirs('%s/EM/PCA' % output_fn_base)
    if not os.path.exists('%s/KMEANS/PCA' % output_fn_base):
        os.makedirs('%s/KMEANS/PCA' % output_fn_base)
    if not os.path.exists('%s/PCADatasets' % output_fn_base):
        os.makedirs('%s/PCADatasets' % output_fn_base)

    print("----Running PCA Experiment-----")
    X = dataset["features"]
    # Standardize data
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(X)

    # Normalizing the Data
    normalized_df = normalize(scaled_df)

    # Converting the numpy array into a pandas DataFrame
    normalized_df = pd.DataFrame(normalized_df)

    # https://www.kaggle.com/vipulgandhi/gaussian-mixture-models-clustering-explained
    # use 2 to get visuals
    pca = PCA(n_components=2)
    new_x = pca.fit_transform(normalized_df)
    new_x = pd.DataFrame(new_x)
    new_x.columns = ['P1', 'P2']

    gm = GaussianMixture(n_components=3)
    km = KMeans(n_clusters=3)
    gm.fit(new_x)
    km.fit(new_x)
    # 3 cluster visual after 2-d PCA reduction
    plt.scatter(new_x['P1'], new_x['P2'],
               c=gm.predict(new_x), cmap=plt.cm.winter, alpha=0.6)
    plt.savefig("./%s/EM/PCA/pca_2_gaussian_%s" % (output_fn_base, output_fn_base))
    plt.close()

    plt.scatter(new_x['P1'], new_x['P2'],
               c=km.predict(new_x), cmap=plt.cm.winter, alpha=0.6)
    plt.savefig("./%s/KMEANS/PCA/pca_2_kmeans_%s" % (output_fn_base, output_fn_base))
    plt.close()

    # metrics for pca 2 and up ##########################################################################################

    main_kmeans_silhouette_scores = []
    main_kmeans_inertias = []
    main_kmeans_fit_times = []
    main_kmeans_ch_scores = []
    main_kmeans_db_scores = []

    main_gm_silhouette_scores = []
    main_gm_fit_times = []
    main_gm_bic_scores = []
    main_gm_aic_scores = []
    main_gm_db_scores = []

    main_pca_fit_times = []
    explained_variances = []
    pca_kmeans_labels_by_dim = []
    pca_gm_labels_by_dim = []

    if(perform_clustering):
        pca = PCA(n_components=best_component_count)
        new_x = pca.fit_transform(normalized_df)

        silhouette_scores = []
        inertias = []
        fit_times = []
        ch_scores = []
        db_scores = []
        kmeans_labels_by_k = []
        # loop over kmeans, this is number 3's experiments
        for k in range(2,max_k_clusters):
            kmeans = KMeans(n_clusters=k)
            t1 = time.time()
            kmeans.fit(new_x)
            fit_times.append(time.time() - t1)
            labels = kmeans.labels_
            ch_scores.append(calinski_harabasz_score(new_x,kmeans.labels_))
            silhouette_scores.append(silhouette_score(new_x,labels))
            inertias.append(kmeans.inertia_)
            db_scores.append(davies_bouldin_score(new_x,labels))
            # save data with features, let clusters be new class column (for part 5):
            cluster_to_features(dataset["class"], labels,
                            "./%s/KMEANS/PCA/KMEANS%s_%d.csv" % (output_fn_base, output_fn_base, k))
            # pd.DataFrame(new_x).assign(dfclass=labels).to_csv(
            #     "./%s/KMEANS/KMEANS%s_%d.csv" % (output_fn_base, output_fn_base, k), index=False)


        main_kmeans_silhouette_scores.append(silhouette_scores)
        main_kmeans_inertias.append(inertias)
        main_kmeans_fit_times.append(fit_times)
        main_kmeans_ch_scores.append(ch_scores)
        main_kmeans_db_scores.append(db_scores)

        silhouette_scores = []
        fit_times = []
        bic_scores = []
        aic_scores = []
        db_scores = []
        gm_labels_by_k = []
        # loop over for gm, this is number 3's experiments
        for component_num in range(2,max_n_components):
            gm = GaussianMixture(n_components=component_num, n_init=10)
            t1 = time.time()
            gm.fit(new_x)
            fit_times.append(time.time() - t1)
            predicted = gm.predict(new_x)

            silhouette_scores.append(silhouette_score(new_x, predicted))
            bic_scores.append(gm.bic(new_x))
            aic_scores.append(gm.aic(new_x))
            db_scores.append(davies_bouldin_score(new_x, predicted))
            gm_labels_by_k.append(predicted)

            if(output_fn_base == "shill" and component_num == 4):
                #pd.DataFrame(predicted).to_csv("./shill_em.csv",index=False)
                lab = pd.DataFrame(predicted).assign(dfclass=dataset["class"]).to_numpy()
                counts = [[],[]]
                for i in range(0,component_num):
                    lab_temp = lab[lab[:, 0] == i]
                    counts[0].append(len(lab_temp[lab_temp[:,1] == 0]))
                    counts[1].append(len(lab_temp[lab_temp[:,1] == 1]))
                pd.DataFrame(np.array(counts)).to_csv("counts_shill_pca_4_em.csv")

            if(output_fn_base == "cardio" and component_num == 4):
                # pd.DataFrame(predicted).to_csv("./cardio_em.csv",index=False)
                lab = pd.DataFrame(predicted).assign(dfclass=dataset["class"]).to_numpy()
                counts = [[], [], [], [], [], [], [], [], [], []]
                for i in range(0, component_num):
                    lab_temp = lab[lab[:, 0] == i]
                    counts[0].append(len(lab_temp[lab_temp[:, 1] == 0]))
                    counts[1].append(len(lab_temp[lab_temp[:, 1] == 1]))
                    counts[2].append(len(lab_temp[lab_temp[:, 1] == 2]))
                    counts[3].append(len(lab_temp[lab_temp[:, 1] == 3]))
                    counts[4].append(len(lab_temp[lab_temp[:, 1] == 4]))
                    counts[5].append(len(lab_temp[lab_temp[:, 1] == 5]))
                    counts[6].append(len(lab_temp[lab_temp[:, 1] == 6]))
                    counts[7].append(len(lab_temp[lab_temp[:, 1] == 7]))
                    counts[8].append(len(lab_temp[lab_temp[:, 1] == 8]))
                    counts[9].append(len(lab_temp[lab_temp[:, 1] == 9]))
                pd.DataFrame(np.array(counts)).to_csv("counts_cardio_pca_4_em.csv")

            # save data with features, let clusters be new class column (for part 5):
            # pd.DataFrame(new_x).assign(dfclass=predicted).to_csv("./%s/EM/EM_%s_%d.csv" % (output_fn_base, output_fn_base, component_num),  index=False)
            cluster_to_features(dataset["class"], predicted,
                            "./%s/EM/PCA/EM%s_%d.csv" % (output_fn_base, output_fn_base, component_num))

        main_gm_aic_scores.append(aic_scores)
        main_gm_bic_scores.append(bic_scores)
        main_gm_db_scores.append(db_scores)
        main_gm_fit_times.append(fit_times)
        main_gm_silhouette_scores.append(silhouette_scores)

        lineplot(x=list(range(2, max_k_clusters)), y=inertias,
                 filename="./%s/KMEANS/PCA/pca_kmeans_%s_%d_dims" % (output_fn_base, output_fn_base, best_component_count),
                 title="KMeans Inertia as K Increases - %s Dim %d" % (output_fn_base, best_component_count), xlab="K",
                 ylab="Inertia", label="Inertia")
        lineplot(x=list(range(2, max_n_components)), y=bic_scores,
                 filename="./%s/EM/PCA/pca_EM_bic_%s_%d_dims" % (output_fn_base, output_fn_base, best_component_count),
                 title="EM BIC as Num Components - %s Dim %d" % (output_fn_base, best_component_count), xlab="Num Components",
                 ylab="BIC", label="BIC")
        lineplot(x=list(range(2, max_n_components)), y=db_scores,
                 filename="./%s/EM/PCA/pca_EM_db_%s_%d_dims" % (output_fn_base, output_fn_base, best_component_count),
                 title="EM Davies-Bouldin as Num Components Increases - %s Dim %d" % (output_fn_base, best_component_count), xlab="Num Components",
                 ylab="DB Score", label="DB Score")

    else:

        for dim in range(2,pca_dim_max):

            pca = PCA(n_components=dim)
            start_pca = time.time()
            new_x = pca.fit_transform(normalized_df)
            main_pca_fit_times.append(time.time() - start_pca)
            new_x = pd.DataFrame(new_x)
            explained_variances.append(pca.explained_variance_ratio_)
            cols = []
            for i in range(0,dim):
                cols.append("P%d" % i)
            new_x.columns = cols

            new_x.assign(dfclass=dataset["class"]).to_csv("./%s/PCADatasets/pca_%s_%d.csv" % (output_fn_base, output_fn_base, dim),  index=False)


        #     silhouette_scores = []
        #     inertias = []
        #     fit_times = []
        #     ch_scores = []
        #     db_scores = []
        #     # loop oover kmeans
        #     kmeans_labels_by_k = []
        #     for k in range(2,max_k_clusters):
        #         kmeans = KMeans(n_clusters=k)
        #         t1 = time.time()
        #         kmeans.fit(new_x)
        #         fit_times.append(time.time() - t1)
        #         labels = kmeans.labels_
        #         ch_scores.append(calinski_harabasz_score(new_x,kmeans.labels_))
        #         silhouette_scores.append(silhouette_score(new_x,labels))
        #         inertias.append(kmeans.inertia_)
        #         db_scores.append(davies_bouldin_score(new_x,labels))
        #         kmeans_labels_by_k.append(labels)
        #     pca_kmeans_labels_by_dim.append(kmeans_labels_by_k)
        #
        #
        #     main_kmeans_silhouette_scores.append(silhouette_scores)
        #     main_kmeans_inertias.append(inertias)
        #     main_kmeans_fit_times.append(fit_times)
        #     main_kmeans_ch_scores.append(ch_scores)
        #     main_kmeans_db_scores.append(db_scores)
        #
        #     silhouette_scores = []
        #     fit_times = []
        #     bic_scores = []
        #     aic_scores = []
        #     db_scores = []
        #     gm_labels_by_k = []
        #     for component_num in range(2,max_n_components):
        #         gm = GaussianMixture(n_components=component_num, n_init=10)
        #         t1 = time.time()
        #         gm.fit(new_x)
        #         fit_times.append(time.time() - t1)
        #         predicted = gm.predict(new_x)
        #
        #         silhouette_scores.append(silhouette_score(new_x, predicted))
        #         bic_scores.append(gm.bic(new_x))
        #         aic_scores.append(gm.aic(new_x))
        #         db_scores.append(davies_bouldin_score(new_x, predicted))
        #         gm_labels_by_k.append(predicted)
        #
        #     pca_gm_labels_by_dim.append(gm_labels_by_k)
        #
        #     main_gm_aic_scores.append(aic_scores)
        #     main_gm_bic_scores.append(bic_scores)
        #     main_gm_db_scores.append(db_scores)
        #     main_gm_fit_times.append(fit_times)
        #     main_gm_silhouette_scores.append(silhouette_scores)
        #
        # # at this point, main_gm_... are arrays of NxM where N is the attempts for PCA dimensions and M is the number of clusters
        # # same goes fo main_kmeans

        # save plot of explained variance as function of n components  #####################################################

        fig, ax1 = plt.subplots()
        plt.bar(list(range(2, pca_dim_max + 1)), list(explained_variances[-1]))
        ax2 = ax1.twinx()

        ax2.plot(list(range(2, pca_dim_max + 1)),list(np.cumsum(pca.explained_variance_ratio_)), 'g-')

        ax1.set_xlabel("Number of Components")
        ax1.set_ylabel("Explained Variance Ratio")
        ax2.set_ylabel("Cumulative Sum", color='g')
        plt.title("PCA Explained Variance as Dimension Parameter Grows - %s" % output_fn_base)
        plt.savefig("./%s/PCADatasets/pca_variance_%s" % (output_fn_base, output_fn_base))
        plt.close()

        # plot entire PCA fit to find elbow ########################################################################
        # this is part of number 2, we want to analyze dim reduction algorithms
        pca = PCA()

        pca.fit(X)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        # Draw lines
        plt.plot(list(range(2, len(cumsum) +2)),cumsum)

        # Create plot
        plt.title("Cumulative Sum of Explained Variance as Dimensions Increase - %s" % output_fn_base)
        plt.xlabel("Dimensions"), plt.ylabel("Explained Variance")
        #plt.tight_layout()
        plt.grid()

        plt.savefig("./%s/PCADatasets/pca_variance_all_%s" % (output_fn_base, output_fn_base))
        plt.close()

        #plot eigenvalues ##########################################################################################
        # this is part of number 2
        # Draw lines
        plt.plot(list(range(2, len(pca.explained_variance_) + 2)), list(pca.explained_variance_))

        # Create plot
        plt.title("Highest Eigenvalues - %s" % output_fn_base)
        plt.xlabel("Dimensions"), plt.ylabel("Eigenvalue")

        plt.grid()

        plt.savefig("./%s/PCADatasets/pca_eigenvalues_all_%s" % (output_fn_base, output_fn_base))
        plt.close()

    metrics_dictionary["pca_gm_labels_dim_by_k"] = pca_gm_labels_by_dim
    metrics_dictionary["pca_kmeans_labels_dim_by_k"] = pca_kmeans_labels_by_dim

    return metrics_dictionary

# Runs The dimentionality reduction algorithms and saves the output
def ica_experiment(dataset, output_fn_base, perform_clustering=False, best_component_count=2):
    metrics_dictionary = {}
    max_k_clusters = 12
    max_n_components = 12
    ica_dim_max = dataset["all"].shape[1]-1
    # Reducing the dimensions of the data
    if not os.path.exists('%s/EM/ICA' % output_fn_base):
        os.makedirs('%s/EM/ICA' % output_fn_base)
    if not os.path.exists('%s/KMEANS/ICA' % output_fn_base):
        os.makedirs('%s/KMEANS/ICA' % output_fn_base)
    if not os.path.exists('%s/ICADatasets' % output_fn_base):
        os.makedirs('%s/ICADatasets' % output_fn_base)

    print("----Running ICA Experiment-----")
    X = dataset["features"]
    # Standardize data
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(X)

    # Normalizing the Data
    normalized_df = normalize(scaled_df)

    # Converting the numpy array into a pandas DataFrame
    normalized_df = pd.DataFrame(normalized_df)

    # use 2 to get visuals
    ica = FastICA(n_components=2)
    new_x = ica.fit_transform(normalized_df)
    new_x = pd.DataFrame(new_x)
    new_x.columns = ['P1', 'P2']

    gm = GaussianMixture(n_components=3)
    km = KMeans(n_clusters=3)
    gm.fit(new_x)
    km.fit(new_x)
    # 3 cluster visual after 2-d ICA reduction
    plt.scatter(new_x['P1'], new_x['P2'],
               c=gm.predict(new_x), cmap=plt.cm.winter, alpha=0.6)
    plt.savefig("./%s/EM/ICA/ica_2_gaussian_%s" % (output_fn_base, output_fn_base))
    plt.close()

    plt.scatter(new_x['P1'], new_x['P2'],
               c=km.predict(new_x), cmap=plt.cm.winter, alpha=0.6)
    plt.savefig("./%s/KMEANS/ICA/ica_2_kmeans_%s" % (output_fn_base, output_fn_base))
    plt.close()

    # metrics for ica 2 and up ##########################################################################################

    main_kmeans_silhouette_scores = []
    main_kmeans_inertias = []
    main_kmeans_fit_times = []
    main_kmeans_ch_scores = []
    main_kmeans_db_scores = []

    main_gm_silhouette_scores = []
    main_gm_fit_times = []
    main_gm_bic_scores = []
    main_gm_aic_scores = []
    main_gm_db_scores = []

    main_ica_fit_times = []
    main_ica_kurtosis = []

    ica_kmeans_labels_by_dim = []
    ica_gm_labels_by_dim = []
    ica_reconstruction = []

    if (perform_clustering):
        ica = FastICA(n_components=best_component_count, max_iter=300)
        new_x = ica.fit_transform(normalized_df)


        silhouette_scores = []
        inertias = []
        fit_times = []
        ch_scores = []
        db_scores = []
        kmeans_labels_by_k = []
        # loop over kmeans, this is number 3's experiments
        for k in range(2,max_k_clusters):
            kmeans = KMeans(n_clusters=k)
            t1 = time.time()
            kmeans.fit(new_x)
            fit_times.append(time.time() - t1)
            labels = kmeans.labels_
            ch_scores.append(calinski_harabasz_score(new_x,kmeans.labels_))
            silhouette_scores.append(silhouette_score(new_x,labels))
            inertias.append(kmeans.inertia_)
            db_scores.append(davies_bouldin_score(new_x,labels))
            # save data with features, let clusters be new class column (for part 5):
            cluster_to_features(dataset["class"], labels,
                            "./%s/KMEANS/ICA/KMEANS%s_%d.csv" % (output_fn_base, output_fn_base, k))
            # pd.DataFrame(new_x).assign(dfclass=labels).to_csv(
            #     "./%s/KMEANS/KMEANS%s_%d.csv" % (output_fn_base, output_fn_base, k), index=False)


        main_kmeans_silhouette_scores.append(silhouette_scores)
        main_kmeans_inertias.append(inertias)
        main_kmeans_fit_times.append(fit_times)
        main_kmeans_ch_scores.append(ch_scores)
        main_kmeans_db_scores.append(db_scores)

        silhouette_scores = []
        fit_times = []
        bic_scores = []
        aic_scores = []
        db_scores = []
        gm_labels_by_k = []
        # loop over for gm, this is number 3's experiments
        for component_num in range(2,max_n_components):
            gm = GaussianMixture(n_components=component_num, n_init=10)
            t1 = time.time()
            gm.fit(new_x)
            fit_times.append(time.time() - t1)
            predicted = gm.predict(new_x)

            silhouette_scores.append(silhouette_score(new_x, predicted))
            bic_scores.append(gm.bic(new_x))
            aic_scores.append(gm.aic(new_x))
            db_scores.append(davies_bouldin_score(new_x, predicted))
            gm_labels_by_k.append(predicted)
            # save data with features, let clusters be new class column (for part 5):
            # pd.DataFrame(new_x).assign(dfclass=predicted).to_csv("./%s/EM/EM_%s_%d.csv" % (output_fn_base, output_fn_base, component_num),  index=False)
            cluster_to_features(dataset["class"], predicted,
                            "./%s/EM/ICA/EM%s_%d.csv" % (output_fn_base, output_fn_base, component_num))

            if(output_fn_base == "shill" and component_num == 8):
                #pd.DataFrame(predicted).to_csv("./shill_em.csv",index=False)
                lab = pd.DataFrame(predicted).assign(dfclass=dataset["class"]).to_numpy()
                counts = [[],[]]
                for i in range(0,component_num):
                    lab_temp = lab[lab[:, 0] == i]
                    counts[0].append(len(lab_temp[lab_temp[:,1] == 0]))
                    counts[1].append(len(lab_temp[lab_temp[:,1] == 1]))
                pd.DataFrame(np.array(counts)).to_csv("counts_shill_ica_8_em.csv")

            if(output_fn_base == "cardio" and component_num == 9):
                # pd.DataFrame(predicted).to_csv("./cardio_em.csv",index=False)
                lab = pd.DataFrame(predicted).assign(dfclass=dataset["class"]).to_numpy()
                counts = [[], [], [], [], [], [], [], [], [], []]
                for i in range(0, component_num):
                    lab_temp = lab[lab[:, 0] == i]
                    counts[0].append(len(lab_temp[lab_temp[:, 1] == 0]))
                    counts[1].append(len(lab_temp[lab_temp[:, 1] == 1]))
                    counts[2].append(len(lab_temp[lab_temp[:, 1] == 2]))
                    counts[3].append(len(lab_temp[lab_temp[:, 1] == 3]))
                    counts[4].append(len(lab_temp[lab_temp[:, 1] == 4]))
                    counts[5].append(len(lab_temp[lab_temp[:, 1] == 5]))
                    counts[6].append(len(lab_temp[lab_temp[:, 1] == 6]))
                    counts[7].append(len(lab_temp[lab_temp[:, 1] == 7]))
                    counts[8].append(len(lab_temp[lab_temp[:, 1] == 8]))
                    counts[9].append(len(lab_temp[lab_temp[:, 1] == 9]))
                pd.DataFrame(np.array(counts)).to_csv("counts_cardio_ica_9_em.csv")

        main_gm_aic_scores.append(aic_scores)
        main_gm_bic_scores.append(bic_scores)
        main_gm_db_scores.append(db_scores)
        main_gm_fit_times.append(fit_times)
        main_gm_silhouette_scores.append(silhouette_scores)

        lineplot(x=list(range(2, max_k_clusters)), y=inertias,
                 filename="./%s/KMEANS/ICA/ica_kmeans_%s_%d_dims" % (output_fn_base, output_fn_base, best_component_count),
                 title="KMeans Inertia as K Increases - %s Dim %d" % (output_fn_base, best_component_count), xlab="K",
                 ylab="Inertia", label="Inertia")
        lineplot(x=list(range(2, max_n_components)), y=bic_scores,
                 filename="./%s/EM/ICA/ica_EM_bic_%s_%d_dims" % (output_fn_base, output_fn_base, best_component_count),
                 title="EM BIC as Num Components - %s Dim %d" % (output_fn_base, best_component_count), xlab="Num Components",
                 ylab="BIC", label="BIC")
        lineplot(x=list(range(2, max_n_components)), y=db_scores,
                 filename="./%s/EM/ICA/ica_EM_db_%s_%d_dims" % (output_fn_base, output_fn_base, best_component_count),
                 title="EM Davies-Bouldin as Num Components Increases - %s Dim %d" % (output_fn_base, best_component_count), xlab="Num Components",
                 ylab="DB Score", label="DB Score")


    else:
        for dim in range(2,ica_dim_max):
            ica = FastICA(n_components=dim,max_iter=300)
            ica_start = time.time()
            new_x = ica.fit_transform(normalized_df)
            main_ica_fit_times.append(time.time() - ica_start)

            # reconstruction error
            pre_img = ica.inverse_transform(new_x)
            ica_reconstruction.append(np.mean((X.to_numpy() - pre_img) ** 2))

            # measure kurtosis on dataset
            new_x = pd.DataFrame(new_x)
            main_ica_kurtosis.append(new_x.kurtosis())


            cols = []
            for i in range(0,dim):
                cols.append("P%d" % i)
            new_x.columns = cols



            # save the dataset for number 4 to rerun the learner
            new_x.assign(dfclass=dataset["class"]).to_csv("./%s/ICADatasets/ica_%s_%d.csv" % (output_fn_base, output_fn_base, dim),  index=False)



        # plot reconstruction error
        # lineplot(x=list(range(2, ica_dim_max)), y=ica_reconstruction,
        #          filename="%s/ICADatasets/reconstruction_error_ica_%s" % (output_fn_base, output_fn_base),
        #          title="Reconstruction Error Component Count Increases - %s" % output_fn_base, xlab="Num Components",
        #          ylab="Reconstruction Err", label="Reconstruction Err")

        avg_kurtosis_with_reconstruction(main_ica_kurtosis, ica_reconstruction, "./%s/ICADatasets/ica_kurt_recon_%s"% (output_fn_base, output_fn_base), "ICA Kurtosis/Reconstruction Error - %s" % dataset_name)
        metrics_dictionary["ica_gm_labels_dim_by_k"] = ica_gm_labels_by_dim
        metrics_dictionary["ica_kmeans_labels_dim_by_k"] = ica_kmeans_labels_by_dim
        metrics_dictionary["ica_kurtosis"] = main_ica_kurtosis

    return metrics_dictionary


def rp_experiment(dataset, output_fn_base, perform_clustering=False, best_component_count = 2):
    metrics_dictionary = {}
    max_k_clusters = 12
    max_n_components = 12
    rp_dim_max = dataset["all"].shape[1]-1
    # Reducing the dimensions of the data
    if not os.path.exists('%s/EM/RP' % output_fn_base):
        os.makedirs('%s/EM/RP' % output_fn_base)
    if not os.path.exists('%s/KMEANS/RP' % output_fn_base):
        os.makedirs('%s/KMEANS/RP' % output_fn_base)
    if not os.path.exists('%s/RPDatasets' % output_fn_base):
        os.makedirs('%s/RPDatasets' % output_fn_base)

    print("----Running Randomized Projections Experiment-----")
    X = dataset["features"]
    # Standardize data
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(X)

    # Normalizing the Data
    normalized_df = normalize(scaled_df)

    # Converting the numpy array into a pandas DataFrame
    normalized_df = pd.DataFrame(normalized_df)

    # https://www.kaggle.com/vipulgandhi/gaussian-mixture-models-clustering-explained
    # use 2 to get visuals
    grp = GaussianRandomProjection(n_components=2)
    new_x = grp.fit_transform(normalized_df)
    new_x = pd.DataFrame(new_x)
    new_x.columns = ['P1', 'P2']

    gm = GaussianMixture(n_components=3)
    km = KMeans(n_clusters=3)
    gm.fit(new_x)
    km.fit(new_x)
    # 3 cluster visual after 2-d RP reduction
    plt.scatter(new_x['P1'], new_x['P2'],
               c=gm.predict(new_x), cmap=plt.cm.winter, alpha=0.6)
    plt.savefig("./%s/EM/RP/rp_2_gaussian_%s" % (output_fn_base, output_fn_base))
    plt.close()

    plt.scatter(new_x['P1'], new_x['P2'],
               c=km.predict(new_x), cmap=plt.cm.winter, alpha=0.6)
    plt.savefig("./%s/KMEANS/RP/rp_2_kmeans_%s" % (output_fn_base, output_fn_base))
    plt.close()


    # multiple attempts to average reconstruction error
    rp_reconstructions = []

    for i in range(0,100):
        rp_reconstruction = []
        for dim in range(2,rp_dim_max):
            rp = GaussianRandomProjection(n_components=dim)
            new_x = rp.fit_transform(normalized_df)
            # reconstruction error
            pre_img = np.dot(new_x, rp.components_) + np.mean(X.to_numpy())
            rp_reconstruction.append(np.mean((X.to_numpy() - pre_img) ** 2))
        rp_reconstructions.append(rp_reconstruction)

    lineplot(x=list(range(2, rp_dim_max)), y=np.array(rp_reconstructions).mean(axis=0),
             filename="%s/RPDatasets/reconstructions_error_rp_%s" % (output_fn_base, output_fn_base),
             title="Reconstruction Error Component Count Increases - %s" % output_fn_base, xlab="Num Components",
             ylab="Reconstruction Err", label="Reconstruction Err")


    # metrics for rp 2 and up ##########################################################################################

    main_kmeans_silhouette_scores = []
    main_kmeans_inertias = []
    main_kmeans_fit_times = []
    main_kmeans_ch_scores = []
    main_kmeans_db_scores = []

    main_gm_silhouette_scores = []
    main_gm_fit_times = []
    main_gm_bic_scores = []
    main_gm_aic_scores = []
    main_gm_db_scores = []

    main_rp_fit_times = []
    main_rp_kurtosis = []
    rp_reconstruction = []
    rp_kmeans_labels_by_dim = []
    rp_gm_labels_by_dim = []

    if (perform_clustering):
        rp = GaussianRandomProjection(n_components=best_component_count)
        new_x = rp.fit_transform(normalized_df)

        silhouette_scores = []
        inertias = []
        fit_times = []
        ch_scores = []
        db_scores = []
        # loop over kmeans, this is number 3's experiments
        for k in range(2,max_k_clusters):
            kmeans = KMeans(n_clusters=k)
            t1 = time.time()
            kmeans.fit(new_x)
            fit_times.append(time.time() - t1)
            labels = kmeans.labels_
            ch_scores.append(calinski_harabasz_score(new_x,kmeans.labels_))
            silhouette_scores.append(silhouette_score(new_x,labels))
            inertias.append(kmeans.inertia_)
            db_scores.append(davies_bouldin_score(new_x,labels))
            # save data with features, let clusters be new class column (for part 5):
            cluster_to_features(dataset["class"], labels,
                                "./%s/KMEANS/RP/KMEANS%s_%d.csv" % (output_fn_base, output_fn_base, k))
            # pd.DataFrame(new_x).assign(dfclass=labels).to_csv(
            #     "./%s/KMEANS/KMEANS%s_%d.csv" % (output_fn_base, output_fn_base, k), index=False)

        main_kmeans_silhouette_scores.append(silhouette_scores)
        main_kmeans_inertias.append(inertias)
        main_kmeans_fit_times.append(fit_times)
        main_kmeans_ch_scores.append(ch_scores)
        main_kmeans_db_scores.append(db_scores)

        silhouette_scores = []
        fit_times = []
        bic_scores = []
        aic_scores = []
        db_scores = []
        gm_labels_by_k = []
        # loop over for gm, this is number 3's experiments
        for component_num in range(2,max_n_components):
            gm = GaussianMixture(n_components=component_num, n_init=10)
            t1 = time.time()
            gm.fit(new_x)
            fit_times.append(time.time() - t1)
            predicted = gm.predict(new_x)

            silhouette_scores.append(silhouette_score(new_x, predicted))
            bic_scores.append(gm.bic(new_x))
            aic_scores.append(gm.aic(new_x))
            db_scores.append(davies_bouldin_score(new_x, predicted))
            gm_labels_by_k.append(predicted)
            # save data with features, let clusters be new class column (for part 5):
            # pd.DataFrame(new_x).assign(dfclass=predicted).to_csv("./%s/EM/EM_%s_%d.csv" % (output_fn_base, output_fn_base, component_num),  index=False)
            cluster_to_features(dataset["class"], predicted,
                            "./%s/EM/RP/EM%s_%d.csv" % (output_fn_base, output_fn_base, component_num))

            if(output_fn_base == "shill" and component_num == 3):
                #pd.DataFrame(predicted).to_csv("./shill_em.csv",index=False)
                lab = pd.DataFrame(predicted).assign(dfclass=dataset["class"]).to_numpy()
                counts = [[],[]]
                for i in range(0,component_num):
                    lab_temp = lab[lab[:, 0] == i]
                    counts[0].append(len(lab_temp[lab_temp[:,1] == 0]))
                    counts[1].append(len(lab_temp[lab_temp[:,1] == 1]))
                pd.DataFrame(np.array(counts)).to_csv("counts_shill_rp_3_em.csv")

            if(output_fn_base == "cardio" and component_num == 5):
                # pd.DataFrame(predicted).to_csv("./cardio_em.csv",index=False)
                lab = pd.DataFrame(predicted).assign(dfclass=dataset["class"]).to_numpy()
                counts = [[], [], [], [], [], [], [], [], [], []]
                for i in range(0, component_num):
                    lab_temp = lab[lab[:, 0] == i]
                    counts[0].append(len(lab_temp[lab_temp[:, 1] == 0]))
                    counts[1].append(len(lab_temp[lab_temp[:, 1] == 1]))
                    counts[2].append(len(lab_temp[lab_temp[:, 1] == 2]))
                    counts[3].append(len(lab_temp[lab_temp[:, 1] == 3]))
                    counts[4].append(len(lab_temp[lab_temp[:, 1] == 4]))
                    counts[5].append(len(lab_temp[lab_temp[:, 1] == 5]))
                    counts[6].append(len(lab_temp[lab_temp[:, 1] == 6]))
                    counts[7].append(len(lab_temp[lab_temp[:, 1] == 7]))
                    counts[8].append(len(lab_temp[lab_temp[:, 1] == 8]))
                    counts[9].append(len(lab_temp[lab_temp[:, 1] == 9]))
                pd.DataFrame(np.array(counts)).to_csv("counts_cardio_rp_5_em.csv")

        main_gm_aic_scores.append(aic_scores)
        main_gm_bic_scores.append(bic_scores)
        main_gm_db_scores.append(db_scores)
        main_gm_fit_times.append(fit_times)
        main_gm_silhouette_scores.append(silhouette_scores)

        lineplot(x=list(range(2, max_k_clusters)), y=inertias,
                 filename="./%s/KMEANS/RP/rp_kmeans_%s_%d_dims" % (output_fn_base, output_fn_base, best_component_count),
                 title="KMeans Inertia as K Increases - %s Dim %d" % (output_fn_base, best_component_count), xlab="K",
                 ylab="Inertia", label="Inertia")
        lineplot(x=list(range(2, max_n_components)), y=bic_scores,
                 filename="./%s/EM/RP/rp_EM_bic_%s_%d_dims" % (output_fn_base, output_fn_base, best_component_count),
                 title="EM BIC as Num Components - %s Dim %d" % (output_fn_base, best_component_count), xlab="Num Components",
                 ylab="BIC", label="BIC")
        lineplot(x=list(range(2, max_n_components)), y=db_scores,
                 filename="./%s/EM/RP/rp_EM_db_%s_%d_dims" % (output_fn_base, output_fn_base, best_component_count),
                 title="EM Davies-Bouldin as Num Components Increases - %s Dim %d" % (output_fn_base, best_component_count), xlab="Num Components",
                 ylab="DB Score", label="DB Score")

    else:
        for dim in range(2,rp_dim_max):
            rp = GaussianRandomProjection(n_components=dim)
            rp_start = time.time()
            new_x = rp.fit_transform(normalized_df)
            main_rp_fit_times.append(time.time()-rp_start)

            # reconstruction error
            pre_img = np.dot(new_x, rp.components_) + np.mean(X.to_numpy())
            rp_reconstruction.append(np.mean((X.to_numpy() - pre_img) ** 2))

            # measure kurtosis on dataset
            new_x = pd.DataFrame(new_x)
            main_rp_kurtosis.append(new_x.kurtosis())


            new_x = pd.DataFrame(new_x)
            cols = []
            for i in range(0,dim):
                cols.append("P%d" % i)
            new_x.columns = cols


            # save the dataset for number 4 to rerun the learner
            new_x.assign(dfclass=dataset["class"]).to_csv("./%s/RPDatasets/rp_%s_%d.csv" % (output_fn_base, output_fn_base, dim),  index=False)

        avg_kurtosis_with_reconstruction(main_rp_kurtosis, rp_reconstruction,
                                         "./%s/RPDatasets/rp_kurt_recon_%s" % (output_fn_base, output_fn_base),
                                         "RP Kurtosis/Reconstruction Error - %s" % output_fn_base)
    plt.close()
    metrics_dictionary["rp_gm_labels_dim_by_k"] = rp_gm_labels_by_dim
    metrics_dictionary["rp_kmeans_labels_dim_by_k"] = rp_kmeans_labels_by_dim

    return metrics_dictionary


def cluster_to_features(class_col, clusters, output):
    pd.DataFrame(np.identity(np.max(clusters) + 1)[clusters]).assign(dfclass=class_col).to_csv(output, index=False)

def fa_experiment(dataset, output_fn_base, perform_clustering=False, return_dataset_from_dim = 2, best_component_count = 2):
    metrics_dictionary = {}
    max_k_clusters = 12
    max_n_components = 12
    fa_dim_max = dataset["all"].shape[1]-1
    # Reducing the dimensions of the data
    if not os.path.exists('%s/EM/FA' % output_fn_base):
        os.makedirs('%s/EM/FA' % output_fn_base)
    if not os.path.exists('%s/KMEANS/FA' % output_fn_base):
        os.makedirs('%s/KMEANS/FA' % output_fn_base)
    if not os.path.exists('%s/FADatasets' % output_fn_base):
        os.makedirs('%s/FADatasets' % output_fn_base)

    print("----Running Factor Analysis Experiment-----")
    X = dataset["features"].astype(np.float64)
    # Standardize data
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(X)

    # Normalizing the Data
    normalized_df = normalize(scaled_df)

    # Converting the numpy array into a pandas DataFrame
    normalized_df = pd.DataFrame(X+0.001*np.random.rand(X.shape[0], X.shape[1])+0.001*np.random.rand(X.shape[0], X.shape[1]))

    # https://www.kaggle.com/vipulgandhi/gaussian-mixture-models-clustering-explained
    # use 2 to get visuals
    fa = FactorAnalysis(n_components=2)
    new_x = fa.fit_transform(normalized_df, y=None)
    new_x = pd.DataFrame(new_x)
    new_x.columns = ['P1', 'P2']

    gm = GaussianMixture(n_components=3)
    km = KMeans(n_clusters=3)
    gm.fit(new_x)
    km.fit(new_x)
    # 3 cluster visual after 2-d fa reduction
    plt.scatter(new_x['P1'], new_x['P2'],
               c=gm.predict(new_x), cmap=plt.cm.winter, alpha=0.6)
    plt.savefig("./%s/EM/FA/fa_2_gaussian_%s" % (output_fn_base, output_fn_base))
    plt.close()

    plt.scatter(new_x['P1'], new_x['P2'],
               c=km.predict(new_x), cmap=plt.cm.winter, alpha=0.6)
    plt.savefig("./%s/KMEANS/FA/fa_2_kmeans_%s" % (output_fn_base, output_fn_base))
    plt.close()

    # metrics for fa 2 and up ##########################################################################################

    main_kmeans_silhouette_scores = []
    main_kmeans_inertias = []
    main_kmeans_fit_times = []
    main_kmeans_ch_scores = []
    main_kmeans_db_scores = []

    main_gm_silhouette_scores = []
    main_gm_fit_times = []
    main_gm_bic_scores = []
    main_gm_aic_scores = []
    main_gm_db_scores = []

    main_fa_fit_times = []
    main_fa_kurtosis = []

    fa_kmeans_labels_by_dim = []
    fa_gm_labels_by_dim = []
    fa_scores = []

    if(perform_clustering):

        fa = fana.factor_analyzer.FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
                                                 method='minres', n_factors=best_component_count, rotation=None, rotation_kwargs={},
                                                 use_smc=True)  # FactorAnalysis(n_components=dim)

        new_x = fa.fit_transform(normalized_df)

        silhouette_scores = []
        inertias = []
        fit_times = []
        ch_scores = []
        db_scores = []
        kmeans_labels_by_k = []
        # loop over kmeans, this is number 3's experiments


        for k in range(2,max_k_clusters):
            kmeans = KMeans(n_clusters=k)
            t1 = time.time()
            kmeans.fit(new_x)
            fit_times.append(time.time() - t1)
            labels = kmeans.labels_
            ch_scores.append(calinski_harabasz_score(new_x,kmeans.labels_))
            silhouette_scores.append(silhouette_score(new_x,labels))
            inertias.append(kmeans.inertia_)
            db_scores.append(davies_bouldin_score(new_x,labels))
            # save data with features, let clusters be new class column (for part 5):
            #pd.DataFrame(new_x).assign(dfclass=labels).to_csv("./%s/KMEANS/KMEANS%s_%d.csv" % (output_fn_base, output_fn_base, k),  index=False)
            cluster_to_features(dataset["class"],labels,"./%s/KMEANS/FA/KMEANS%s_%d.csv" % (output_fn_base, output_fn_base, k))
        fa_kmeans_labels_by_dim.append(kmeans_labels_by_k)





        main_kmeans_silhouette_scores.append(silhouette_scores)
        main_kmeans_inertias.append(inertias)
        main_kmeans_fit_times.append(fit_times)
        main_kmeans_ch_scores.append(ch_scores)
        main_kmeans_db_scores.append(db_scores)

        silhouette_scores = []
        fit_times = []
        bic_scores = []
        aic_scores = []
        db_scores = []
        gm_labels_by_k = []
        # loop over for gm, this is number 3's experiments
        for component_num in range(2,max_n_components):
            gm = GaussianMixture(n_components=component_num, n_init=10)
            t1 = time.time()
            gm.fit(new_x)
            fit_times.append(time.time() - t1)
            predicted = gm.predict(new_x)

            silhouette_scores.append(silhouette_score(new_x, predicted))
            bic_scores.append(gm.bic(new_x))
            aic_scores.append(gm.aic(new_x))
            db_scores.append(davies_bouldin_score(new_x, predicted))
            gm_labels_by_k.append(predicted)
            # save data with features, let clusters be new class column (for part 5):
            cluster_to_features(dataset["class"], predicted,
                                "./%s/EM/FA/EM%s_%d.csv" % (output_fn_base, output_fn_base, component_num))
            #pd.DataFrame(new_x).assign(dfclass=predicted).to_csv("./%s/EM/EM_%s_%d.csv" % (output_fn_base, output_fn_base, component_num),  index=False)
            if(output_fn_base == "shill" and component_num == 5):
                #pd.DataFrame(predicted).to_csv("./shill_em.csv",index=False)
                lab = pd.DataFrame(predicted).assign(dfclass=dataset["class"]).to_numpy()
                counts = [[],[]]
                for i in range(0,component_num):
                    lab_temp = lab[lab[:, 0] == i]
                    counts[0].append(len(lab_temp[lab_temp[:,1] == 0]))
                    counts[1].append(len(lab_temp[lab_temp[:,1] == 1]))
                pd.DataFrame(np.array(counts)).to_csv("counts_shill_fa_5_em.csv")

            if(output_fn_base == "cardio" and component_num == 5):
                # pd.DataFrame(predicted).to_csv("./cardio_em.csv",index=False)
                lab = pd.DataFrame(predicted).assign(dfclass=dataset["class"]).to_numpy()
                counts = [[], [], [], [], [], [], [], [], [], []]
                for i in range(0, component_num):
                    lab_temp = lab[lab[:, 0] == i]
                    counts[0].append(len(lab_temp[lab_temp[:, 1] == 0]))
                    counts[1].append(len(lab_temp[lab_temp[:, 1] == 1]))
                    counts[2].append(len(lab_temp[lab_temp[:, 1] == 2]))
                    counts[3].append(len(lab_temp[lab_temp[:, 1] == 3]))
                    counts[4].append(len(lab_temp[lab_temp[:, 1] == 4]))
                    counts[5].append(len(lab_temp[lab_temp[:, 1] == 5]))
                    counts[6].append(len(lab_temp[lab_temp[:, 1] == 6]))
                    counts[7].append(len(lab_temp[lab_temp[:, 1] == 7]))
                    counts[8].append(len(lab_temp[lab_temp[:, 1] == 8]))
                    counts[9].append(len(lab_temp[lab_temp[:, 1] == 9]))
                pd.DataFrame(np.array(counts)).to_csv("counts_cardio_fa_5_em.csv")
        fa_gm_labels_by_dim.append(gm_labels_by_k)

        main_gm_aic_scores.append(aic_scores)
        main_gm_bic_scores.append(bic_scores)
        main_gm_db_scores.append(db_scores)
        main_gm_fit_times.append(fit_times)
        main_gm_silhouette_scores.append(silhouette_scores)
        # plot_scores('%s/EM/FA' % output_fn_base, aic=main_gm_aic_scores, bic=main_gm_bic_scores, db=main_gm_db_scores, fit=main_gm_fit_times, silhouette=main_gm_silhouette_scores, clustering_method="GM", decomp_method="FA")
        # plot_scores('%s/KMEANS/FA' % output_fn_base, silhouette=main_kmeans_silhouette_scores, inertias= main_kmeans_inertias, fit=main_kmeans_fit_times, ch=main_kmeans_ch_scores, db=main_kmeans_db_scores,clustering_method="KMeans", decomp_method="FA")
        metrics_dictionary["fa_gm_labels_dim_by_k"] = fa_gm_labels_by_dim
        metrics_dictionary["fa_kmeans_labels_dim_by_k"] = fa_kmeans_labels_by_dim
        lineplot(x=list(range(2, max_k_clusters)), y=inertias,
                 filename="./%s/KMEANS/FA/fa_kmeans_%s_%d_dims" % (output_fn_base, output_fn_base, best_component_count),
                 title="KMeans Inertia as K Increases - %s Dim %d" % (output_fn_base, best_component_count), xlab="K",
                 ylab="Inertia", label="Inertia")
        lineplot(x=list(range(2, max_n_components)), y=bic_scores,
                 filename="./%s/EM/FA/fa_EM_bic_%s_%d_dims" % (output_fn_base, output_fn_base, best_component_count),
                 title="EM BIC as Num Components - %s Dim %d" % (output_fn_base, best_component_count), xlab="Num Components",
                 ylab="BIC", label="BIC")
        lineplot(x=list(range(2, max_n_components)), y=db_scores,
                 filename="./%s/EM/FA/fa_EM_db_%s_%d_dims" % (output_fn_base, output_fn_base, best_component_count),
                 title="EM Davies-Bouldin as Num Components Increases - %s Dim %d" % (output_fn_base, best_component_count), xlab="Num Components",
                 ylab="DB Score", label="DB Score")
    else:
        for dim in range(2, fa_dim_max):
            fa = fana.factor_analyzer.FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
                                                     method='minres', n_factors=dim, rotation=None, rotation_kwargs={},
                                                     use_smc=True)  # FactorAnalysis(n_components=dim)

            fa_start = time.time()
            new_x = fa.fit_transform(normalized_df)
            # measure kurtosis on dataset
            main_fa_kurtosis.append([])
            main_fa_fit_times.append(time.time() - fa_start)

            # fa_scores.append(np.mean(cross_val_score(fa, X.astype(np.float64))))

            new_x = pd.DataFrame(new_x)
            cols = []
            for i in range(0, dim):
                cols.append("P%d" % i)
            new_x.columns = cols

            # save the dataset for number 4 to rerun the learner
            new_x.assign(dfclass=dataset["class"]).to_csv(
                "./%s/FADatasets/fa_%s_%d.csv" % (output_fn_base, output_fn_base, dim), index=False)

            if (dim == return_dataset_from_dim):
                metrics_dictionary["returned"] = new_x.assign(dfclass=dataset["class"])
    lineplot(x=list(range(2, fa_dim_max)), y=np.array(fa.get_eigenvalues()[0][2:]),
             filename="%s/FADatasets/FA_eigen_%s" % (output_fn_base, output_fn_base),
             title="FA Eigenvalues as Component Count Increases - %s" % output_fn_base, xlab="Num Components",
             ylab="Eigenvalues", label="Eigenvalues")
    return metrics_dictionary


"""
Compare Clusters Function

kmeans_cluster_list: a list of cluster label arrays, the list is of length k, k is the max number of clusters found in the kmeans experiment
[ k1 ... [ n1 ... ]]

gm_cluster_list: a list of cluster label arrays, the list is of length k, k is the max number of clusters found in the gm/em experiment
pca_kmeans_cluster_matrix, pca_gm_cluster_matrix ,ica_kmeans_cluster_matrix, ica_gm_cluster_matrix: a list of lists of arrays of cluster values. 
Essentially it is like running the kmeans and gm experiments multiple times and creating an array out of their lists of clusters, so this is dimxkxn where dim
is the maximum number of dimensions tested in the pca experiment, k is the maximumn number of clusters in each of these experiments,
and n is just the number of datapoints: [ dim1 ... [ k1 ... [ n1 ... ]]
"""
def compare_clusters(kmeans_cluster_list, gm_cluster_list, pca_kmeans_cluster_matrix, pca_gm_cluster_matrix, ica_kmeans_cluster_matrix, ica_gm_cluster_matrix):
    # todo finish this function when other parts implemented
    return

def plot_scores(output_folder, aic= None, bic = None,db = None, fit = None, silhouette = None, ch = None, inertias=None, clustering_method = "KMeans", decomp_method="PCA", combine=False, k=3):
    # aic #############
    if(aic is not None):
        aic = np.array(aic)
        plt.plot(x = list(range(2, len(aic))), y =aic[:,k-2], linewidth=2, label="AIC - %s - k=%d" % (clustering_method, k))

        if not combine:
            plt.title("AIC Measure - %s Clustering (K=%d)- %s As Number of Components Increases" % (clustering_method,k,decomp_method))
            plt.xlabel("Number of Components"), plt.ylabel("Value")
            plt.grid()
            plt.savefig("%s/aic_%s_%s_k_%d" % (output_folder, decomp_method, clustering_method,k))
            plt.close()
    # bic #############
    if(bic is not None):
        bic = np.array(bic)
        plt.plot(bic[:,k-2], linewidth=2, label="BIC - %s - k=%d" % (clustering_method, k))

        if not combine:
            plt.title("BIC Measure - %s Clustering (K=%d)- %s As Number of Components Increases" % (clustering_method,k,decomp_method))
            plt.xlabel("Number of Components"), plt.ylabel("Value")
            plt.grid()
            plt.savefig("%s/bic_%s_%s_k_%d" % (output_folder, decomp_method, clustering_method,k))
            plt.close()
    return

def kmeans_em_no_dim_red_compare(kmeans_metrics, em_metrics, dataset_name, output):
    x = kmeans_metrics["cluster_x_tics"]

    fontsize = 15
    # kmeans_metrics["kmeans_silhouette"]
    # kmeans_metrics["kmeans_db"]
    #
    # em_metrics["em_silhouette"] = silhouette_scores
    # em_metrics["em_db"] = db_scores

    # Runtime
    plt.plot(x,kmeans_metrics["kmeans_fit_times"], linewidth=3, label="kmeans", color="purple")
    plt.plot(x, em_metrics["em_fit_times"], linewidth=3, label="em", color="black")
    plt.title("Fit Runtime As Number of Clusters Increases - %s Dataset" % dataset_name, fontsize=fontsize)
    plt.xlabel("Number of Clusters", fontsize=fontsize), plt.ylabel("Runtime", fontsize=fontsize)
    plt.tight_layout()
    plt.grid()
    plt.xticks(range(x[0], x[-1]+1,2))
    plt.legend(fontsize=fontsize)

    plt.savefig("./%s/em_kmeans_fittimes_%s" % (output, dataset_name))
    plt.close()

    # silhouette
    plt.plot(x,kmeans_metrics["kmeans_silhouette"], linewidth=3, label="kmeans", color="purple")
    plt.plot(x, em_metrics["em_silhouette"], linewidth=3, label="em", color="black")
    plt.title("Silhouette Score As Number of Clusters Increases - %s Dataset" % dataset_name, fontsize=fontsize)
    plt.xlabel("Number of Clusters", fontsize=fontsize), plt.ylabel("Runtime", fontsize=fontsize)
    plt.tight_layout()
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.xticks(range(x[0], x[-1]+1,2))

    plt.savefig("./%s/em_kmeans_silhouette_%s" % (output, dataset_name))
    plt.close()

    #db
    plt.plot(x,kmeans_metrics["kmeans_db"], linewidth=3, label="kmeans", color="purple")
    plt.plot(x, em_metrics["em_db"], linewidth=3, label="em", color="black")
    plt.title("Davies Bouldin Score As Number of Clusters Increases - %s Dataset" % dataset_name, fontsize=fontsize)
    plt.xlabel("Number of Clusters", fontsize=fontsize), plt.ylabel("Runtime", fontsize=fontsize)
    plt.tight_layout()
    plt.legend(fontsize=fontsize)
    plt.grid()
    plt.xticks(range(x[0], x[-1]+1,2))

    plt.savefig("./%s/em_kmeans_db_%s" % (output, dataset_name))
    plt.close()

def shill_cardio_cluster_no_dim_compare(shill_kmeans_metrics, shill_em_metrics, cardio_kmeans_metrics, cardio_em_metrics, output):
    x = shill_kmeans_metrics["cluster_x_tics"]

    fontsize = 15
    # kmeans_metrics["kmeans_silhouette"]
    # kmeans_metrics["kmeans_db"]
    #
    # em_metrics["em_silhouette"] = silhouette_scores
    # em_metrics["em_db"] = db_scores

    # Kmeans Inertia (for elbow)
    plt.plot(x,shill_kmeans_metrics["inertia"], linewidth=3, label="Shill Dataset Inertia", color="green")
    plt.plot(x, cardio_kmeans_metrics["inertia"], linewidth=3, label="Cardio Dataset Inertia", color="blue")
    plt.title("KMeans Inertia As K Increases - Both Datasets", fontsize=fontsize)
    plt.xlabel("Number of Clusters", fontsize=fontsize), plt.ylabel("Inertia", fontsize=fontsize)
    plt.tight_layout()
    plt.grid()
    plt.xticks(range(x[0], x[-1]+1,2))
    plt.legend(fontsize=fontsize)

    plt.savefig("./kmeans_inertias_%s" % (output))
    plt.close()

#def create_pairwise_plots(X):

def create_tsne(X, labels, output, perplexity=10):
    tsne = TSNE(n_components=2, perplexity=perplexity).fit_transform(X)

    plt.scatter(tsne[:,0], tsne[:,1],
               c=labels, cmap=plt.cm.winter, alpha=0.6)
    plt.savefig("./%s" % (output))
    plt.close()

def avg_kurtosis(X,output, title):
    fontsize = 15
    # convert to numpy array
    h = len(X)
    w = len(X[-1])
    kurtosis_np = np.full([h, w], np.nan)
    for j, arr in enumerate(X):
        for i, ele in enumerate(arr):
            kurtosis_np[j,i] = ele
    np.nanmean(kurtosis_np, axis=1)
    plt.plot(range(2,len(X)+2),np.nanmean(kurtosis_np, axis=1), linewidth=3, label="AVG Kurtosis", color="green")
    plt.title(title, fontsize=fontsize)
    plt.xlabel("Number of Components", fontsize=fontsize), plt.ylabel("Kurtosis", fontsize=fontsize)
    plt.tight_layout()
    plt.grid()

    plt.savefig(output)
    plt.close()

def avg_kurtosis_with_reconstruction(kurt, reconstruction, output, title):

    fontsize = 15
    # convert to numpy array
    h = len(kurt)
    w = len(kurt[-1])
    kurtosis_np = np.full([h, w], np.nan)
    for j, arr in enumerate(kurt):
        for i, ele in enumerate(arr):
            kurtosis_np[j,i] = ele
    np.nanmean(kurtosis_np, axis=1)

    fig, ax1 = plt.subplots()
    ax1.plot(range(2,len(kurt)+2),reconstruction, 'b-')
    ax2 = ax1.twinx()

    ax2.plot(list(range(2,len(kurt)+2)),np.nanmean(kurtosis_np, axis=1), 'r-')

    ax1.set_xlabel("Number of Components", fontsize=fontsize)
    ax1.set_ylabel("Reconstruction Error", color='blue', fontsize=fontsize)
    ax2.set_ylabel("AVG Kurtosis", color='red', fontsize=fontsize)

    plt.title(title, fontsize=fontsize)
    plt.savefig(output)
    plt.close()

def cluster(X, output_fn_base, em_com=2, k_k=2):
    # dim+cluseter function

    kmeans = KMeans(n_clusters=k_k)
    t1 = time.time()
    kmeans.fit(X)
    fit_time = (time.time() - t1)
    labels = kmeans.labels_
    ch_score = calinski_harabasz_score(X,kmeans.labels_)
    silhouette_sc = silhouette_score(X,labels)
    inertia= kmeans.inertia_
    db_score = davies_bouldin_score(X,labels)
    cluster_labels= labels # used to handle the following:
    # When you reproduced your clustering experiments on the datasets projected onto the new spaces created by ICA,
    # PCA, and RP, did you get the same clusters as before? Different clusters? Why? Why not?

    metric_kmeans = {}
    metric_kmeans["ch_score"] = ch_score
    metric_kmeans["fit_time"] = fit_time
    metric_kmeans["silhouette_score"] = silhouette_sc
    metric_kmeans["inertia"] = inertia
    metric_kmeans["db_score"] = db_score
    metric_kmeans["labels"] = cluster_labels


    #EM

    gm = GaussianMixture(n_components=em_com, n_init=10)
    t1 = time.time()
    gm.fit(X)
    fit_time = time.time() - t1
    predicted = gm.predict(X)
    ch_score = calinski_harabasz_score(X, predicted)
    silhouette_sc = silhouette_score(X, predicted)
    bic_score= gm.bic(X)
    aic_score = gm.aic(X)
    db_score = davies_bouldin_score(X, predicted)

    metric_em = {}
    metric_em["ch_score"] = ch_score
    metric_em["fit_time"] = fit_time
    metric_em["silhouette_score"] = silhouette_sc
    metric_em["bic_score"] = bic_score
    metric_em["db_score"] = db_score
    metric_em["labels"] = predicted
    metric_em["aic_score"] = aic_score

    return metric_kmeans, metric_em

if __name__ == "__main__":
    print("#####################################################################")
    print("CS7641 ML - Un-Supervised Learning Assignment Test Program")
    print("#####################################################################")
    datasets, filepaths = get_datasets()

    # Writing to file
    with open("metrics.txt", "w") as file1:
        file1.write("CS7641 ML - Un-Supervised Learning Assignment Test Program \n")
        file1.write("Derek Chase Brown (dbrown381@gatech.edu) \n")
        file1.write("============ Datasets ============ \n")
        file1.writelines(filepaths)
        file1.write("\n============ Metrics ============ \n\n")
        test_dict = {}
        # Test workbenches:
        for dataset_name in datasets:
            test_dict[dataset_name] = {}
            file1.write("DATASET: %s\n---------------------------------------------------------\n\n" % dataset_name)
            print("Testing on dataset: %s" % dataset_name )
            if not os.path.exists('%s/KMEANS' % dataset_name):
                os.makedirs('%s/KMEANS' % dataset_name)
            if not os.path.exists('%s/EM' % dataset_name):
                os.makedirs('%s/EM' % dataset_name)

            # no dimension changes ############################################################################
            no_dim_kmeans_logs, no_dim_kmeans_metrics =kmeans_experiment(datasets[dataset_name],get_hyperparams(dataset_name,"KMeans"), dataset_name)
            no_dim_em_logs, no_dim_em_metrics = em_experiment(datasets[dataset_name],
                                                        get_hyperparams(dataset_name, "EM"), dataset_name)
            
            test_dict[dataset_name]["kmeans_nodim_metrics"] = no_dim_kmeans_metrics
            test_dict[dataset_name]["em_nodim_metrics"] = no_dim_em_metrics

            kmeans_em_no_dim_red_compare(kmeans_metrics=no_dim_kmeans_metrics, em_metrics=no_dim_em_metrics, dataset_name=dataset_name, output='%s/' % dataset_name)

            ###################################################################################################
            # only dim, no cluster
            pca_experiment(datasets[dataset_name], dataset_name)
            ica_metrics = ica_experiment(datasets[dataset_name], dataset_name)
            rp_experiment(datasets[dataset_name], dataset_name)
            fa_metrics = fa_experiment(datasets[dataset_name], dataset_name, return_dataset_from_dim= 3 if dataset_name == "shill" else 8)

            kurt = ica_metrics["ica_kurtosis"]
            # plot kurtosis
            avg_kurtosis(kurt,"./%s_kurtosis_ica" %dataset_name, "ICA Kurtosis - %s Dataset" % dataset_name)
            plt.close()

            ###################################################################################################
            # dim and cluster

            fa_experiment(datasets[dataset_name], dataset_name,
                                       return_dataset_from_dim=3 if dataset_name == "shill" else 8,
                                       best_component_count= 3 if dataset_name == "shill" else 8, perform_clustering=True)
            ica_experiment(datasets[dataset_name], dataset_name, best_component_count=4 if dataset_name == "shill" else 10, perform_clustering=True)
            rp_experiment(datasets[dataset_name], dataset_name, best_component_count= 4 if dataset_name == "shill" else 7, perform_clustering=True)
            pca_experiment(datasets[dataset_name], dataset_name, perform_clustering=True, best_component_count=7 if dataset_name == "shill" else 17)

        # post analysis
        # no dim:
        shill_cardio_cluster_no_dim_compare(test_dict["shill"]["kmeans_nodim_metrics"],None,test_dict["cardio"]["kmeans_nodim_metrics"],None,"nodim")
