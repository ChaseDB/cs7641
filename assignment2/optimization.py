import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import six
import sys
#sys.modules['sklearn.externals.six'] = six
#import mlrose as mlrose
import mlrose_hiive as mlrose
import time
from pandas import read_csv
import pandas as pd

from sklearn.model_selection import learning_curve, cross_val_score, train_test_split, ShuffleSplit, cross_val_predict
import pydot
import matplotlib
import matplotlib.pyplot as plt

import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier

# the following line might need to be commented out on some machines
matplotlib.use("TkAgg")

HYPERPARAMS = {
    "tsp":
    {
        "Hill": {
            "max_iters": 250,
            "restarts": 10,
            "init_state": None
        },
        "SA": {
            "max_iters": 250,
            "decay": "Geometric",
            "max_attempts" : 10,
            "initial_temp": 20
        },
        "GA": {
            "pop_size" : 200,
            "mutation_prob" : .4,
            "max_attempts": 10,
            "max_iters": 250
        },
        "MIMIC": {
            "pop_size": 100,
            "keep_pct": .5,
            "max_attempts": 5,
            "max_iters": 250,
            "fast_mimic": False
        }
    },
    "four_peaks":
        {
            "Hill": {
                "max_iters": 800,
                "restarts": 10,
                # "init_state": [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
            },
            "SA": {
                "max_iters": 800,
                "decay": "Geometric",
                "max_attempts": 5,
                "initial_temp": 200
            },
            "GA": {
                "pop_size": 200,
                "mutation_prob": .4,
                "max_attempts": 10,
                "max_iters": 800
            },
            "MIMIC": {
                "pop_size": 100,
                "keep_pct": .5,
                "max_attempts": 5,
                "max_iters": 800,
            }
        },
    "one_max":
        {
            "Hill": {
                "max_iters": 250,
                "restarts": 100,
                # "init_state": [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
            },
            "SA": {
                "max_iters": 250,
                "decay": "Geometric",
                "max_attempts": 10,
                "initial_temp": 20
            },
            "GA": {
                "pop_size": 200,
                "mutation_prob": .2,
                "max_attempts": 10,
                "max_iters": 250
            },
            "MIMIC": {
                "pop_size": 100,
                "keep_pct": .5,
                "max_attempts": 5,
                "max_iters": 250,
            }
        },
}

METRICSSTORE = []

def get_fitness_functions():
    df = pd.read_csv("./houston2008_order.csv")
    coord_list = list(df[['lat', 'long']].apply(tuple, axis=1))
    coord_list = coord_list[0:30]
    fitness_tsp = mlrose.TravellingSales(coords=coord_list)
    problem_tsp = mlrose.TSPOpt(length=len(coord_list), fitness_fn=fitness_tsp, maximize=False)

    fitness_fourpeak = mlrose.FourPeaks(t_pct=.3)
    problem_fourpeak = mlrose.DiscreteOpt(length=20,fitness_fn=fitness_fourpeak)

    fitness_flipflop = mlrose.FlipFlop()
    problem_flipflop = mlrose.DiscreteOpt(length=30,fitness_fn=fitness_flipflop)

    fitness_one_max = mlrose.OneMax()
    problem_one_max = mlrose.DiscreteOpt(length=35,fitness_fn=fitness_one_max,)

    weights = [10, 5, 2, 8, 15]
    values = [1, 2, 3, 4, 5]
    max_weight_pct = 0.6
    fitness_knapsack = mlrose.Knapsack(weights, values, max_weight_pct)
    problem_knapsack = mlrose.DiscreteOpt(length=5,fitness_fn=fitness_knapsack)

    return {
                "tsp": problem_tsp,
                "four_peaks": problem_fourpeak,
                "one_max": problem_one_max,
           }

def get_hyperparams(dataset_name, algorithm):
    return HYPERPARAMS[dataset_name][algorithm]

def plot_single(data, fn, title="Fitness Score Based On Number of Iterations", xlab="Iterations Taken", xvals = None, datalabel = "Fitness", ylab="Fitness"):
    # Draw lines
    if(xvals is None):
        plt.plot(data, linewidth=2, label=datalabel)
    else:
        plt.plot(xvals,data, linewidth=2, label=datalabel)

    # Create plot
    plt.title(title)
    plt.xlabel(xlab), plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig("./%s" % fn)
    plt.close()

def plot_multiple(data, labels, fn, title="Fitness Score Based On Number of Iterations", xlab="Iterations Taken", ylab="Fitness", xvals = None):
    # Draw lines
    for i , series in enumerate(data):
        if(xvals is None):
            plt.plot(series, linewidth=2, label=labels[i])
        else:
            plt.plot(xvals,series, linewidth=2, label=labels[i])

    # Create plot
    plt.title(title)
    plt.xlabel(xlab), plt.ylabel(ylab)
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("./%s" % fn)
    plt.close()

def plot_runtimes(runtimes, fn, xlab="Iteration Number", title=None):
    # Draw lines
    plt.plot(runtimes, linewidth=2, label="Runtime")

    # Create plot
    if(title is None):
        plt.title("Runtime Over Iterations - %s" % fn)
    else:
        plt.title(title)
    plt.xlabel(xlab), plt.ylabel("Time")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("./%s" % fn)
    plt.close()

def hill_climb_experiment(optimization_problem, hparams, output_fn_base):
    metrics = {}
    logs = []
    print("----Running Hill Climbing Experiment-----")
    print("Hyperparameters Used: ")
    print(hparams)

    best_state, best_fitness, curve = mlrose.random_hill_climb(optimization_problem,max_iters=hparams["max_iters"],restarts=hparams["restarts"],curve=True)
    curve = -curve
    # Iterations
    fitness_scores = []
    runtimes = []
    iteration_count = range(1,hparams["max_iters"])
    for iter in iteration_count:
        start_time = time.time()
        best_state, best_fitness, _ = mlrose.random_hill_climb(optimization_problem, max_iters= iter, restarts=hparams["restarts"])
        end_time = time.time()
        runtimes.append(end_time - start_time)
        fitness_scores.append(-best_fitness if optimization_problem.prob_type == 'tsp' else best_fitness)

    plot_single(fitness_scores,"%s/Hill/hillclimb_fitness_iterations" % output_fn_base, xvals=iteration_count)
    plot_runtimes(runtimes,"%s/Hill/hillclimb_runtime_iterations" % output_fn_base)

    metrics["runtimes"] = runtimes
    metrics["fitness"] = fitness_scores


    # restarts
    fitness_scores = []
    restarts = range(1,300)
    runtimes = []
    for rest in restarts:
        start_time = time.time()
        best_state, best_fitness, curve = mlrose.random_hill_climb(optimization_problem, max_iters=hparams["max_iters"],
                                                                   restarts=rest, curve=True)
        end_time = time.time()
        runtimes.append(end_time - start_time)
        fitness_scores.append(-best_fitness if optimization_problem.prob_type == 'tsp' else best_fitness)

    plot_single(fitness_scores,"%s/Hill/restarts" % output_fn_base,title="Randomized Hill Climb Fitness As Restart Count Grows - %s" % get_name_of_experiment(output_fn_base), xlab="Number of Restarts",xvals=restarts)
    plot_runtimes(runtimes, "%s/Hill/hillclimb_runtime_restarts" % output_fn_base,"Restarts", "Runtime as Restarts Grows - %s" % get_name_of_experiment(output_fn_base))
    metrics["runtimes-hill"] = runtimes
    logs.append("\tHyperparameters: \n")
    logs.append("\t%s" % str(hparams))
    logs.append("\n\n\tBest State: \n\t\t")
    logs.append(str(list(best_state)))
    logs.append("\n\tBest Fitness: \n\t\t")
    logs.append(str(best_fitness))


    return logs, metrics

def simulated_annealing_experiment(optimization_problem, hparams, output_fn_base):
    metrics = {}
    logs = []
    print("----Running Simulated Annealing Experiment-----")
    print("Hyperparameters Used: ")
    print(hparams)

    schedule = None
    if(hparams["decay"] == "Geometric"):
        schedule = mlrose.GeomDecay(init_temp=hparams["initial_temp"])
    elif(hparams["decay"] == "Arithmetic"):
        schedule = mlrose.ArithDecay(init_temp=hparams["initial_temp"])
    else:
        schedule = mlrose.ExpDecay(init_temp=hparams["initial_temp"])

    best_state, best_fitness, _ = mlrose.simulated_annealing(optimization_problem,schedule,hparams["max_attempts"], hparams["max_iters"])

    # Iterations and runtime
    fitness_scores = []
    runtimes = []
    iteration_count = range(1,hparams["max_iters"])
    for iter in iteration_count:
        start_time = time.time()
        best_state, best_fitness, _ = mlrose.simulated_annealing(optimization_problem,schedule,hparams["max_attempts"], iter)
        end_time = time.time()
        runtimes.append(end_time - start_time)
        fitness_scores.append(-best_fitness if optimization_problem.prob_type == 'tsp' else best_fitness)

    plot_single(fitness_scores,"%s/Annealing/annealing_fitness_iterations" % output_fn_base, xvals=iteration_count)
    plot_runtimes(runtimes,"%s/Annealing/annealing_runtime_iterations" % output_fn_base)
    metrics["runtimes"] = runtimes
    metrics["fitness"] = fitness_scores

    # intial temp
    fitness_scores_geo = []
    fitness_scores_arith = []
    fitness_scores_exp = []
    temps = range(1,100)

    for temp in temps:
        decays = [mlrose.GeomDecay(init_temp=temp),
                  mlrose.ArithDecay(init_temp=temp),
                  mlrose.ExpDecay(init_temp=temp)]
        for i in range(0,len(decays)):
            best_state, best_fitness, _ = mlrose.simulated_annealing(optimization_problem,decays[i],hparams["max_attempts"],400)
            if(i == 0):
                fitness_scores_geo.append(-best_fitness if optimization_problem.prob_type == 'tsp' else best_fitness)
            elif(i==1):
                fitness_scores_arith.append(-best_fitness if optimization_problem.prob_type == 'tsp' else best_fitness)
            else:
                fitness_scores_exp.append(-best_fitness if optimization_problem.prob_type == 'tsp' else best_fitness)


    # Iterations and runtime
    decays = [mlrose.GeomDecay(init_temp=temp),
              mlrose.ArithDecay(init_temp=temp),
              mlrose.ExpDecay(init_temp=temp)]
    fitness_scores_geo2 = []
    fitness_scores_arith2 = []
    fitness_scores_exp2 = []
    for i in range(0, len(decays)):
        if (i == 0):
            iteration_count2 = range(1,2* hparams["max_iters"])
            for iter in iteration_count2:
                best_state, best_fitness, _ = mlrose.simulated_annealing(optimization_problem, decays[i],
                                                                         hparams["max_attempts"], iter)
                fitness_scores_geo2.append(-best_fitness if optimization_problem.prob_type == 'tsp' else best_fitness)
        elif (i == 1):
            iteration_count2 = range(1, 2*hparams["max_iters"])
            for iter in iteration_count2:
                best_state, best_fitness, _ = mlrose.simulated_annealing(optimization_problem, decays[i],
                                                                         hparams["max_attempts"], iter)
                fitness_scores_arith2.append(-best_fitness if optimization_problem.prob_type == 'tsp' else best_fitness)
        else:
            iteration_count2 = range(1, 2*hparams["max_iters"])
            for iter in iteration_count2:

                best_state, best_fitness, _ = mlrose.simulated_annealing(optimization_problem, decays[i],
                                                                         hparams["max_attempts"], iter)
                fitness_scores_exp2.append(-best_fitness if optimization_problem.prob_type == 'tsp' else best_fitness)



    plot_single(fitness_scores,"%s/Annealing/annealing_fitness_iterations" % output_fn_base, xvals=iteration_count)
    plot_runtimes(runtimes,"%s/Annealing/annealing_runtime_iterations" % output_fn_base)
    metrics["runtimes"] = runtimes
    metrics["fitness"] = fitness_scores


    plot_multiple([fitness_scores_geo2, fitness_scores_arith2, fitness_scores_exp2],
                   ["Geometric Decay Fitness", "Arithmetic Decay Fitness", "Exponential Decay Fitness"],
                   "%s/Annealing/sa_fitness_decay_iterations" % output_fn_base,
                   title="SA Fitness Scores Over Iterations - %s" % get_name_of_experiment(output_fn_base), xlab="Iterations")

    plot_multiple([fitness_scores_geo, fitness_scores_arith, fitness_scores_exp],
                   ["Geometric Decay Fitness", "Arithmetic Decay Fitness", "Exponential Decay Fitness"],
                   "%s/Annealing/sa_fitness_decay" % output_fn_base,
                   title="SA Fitness Scores Over Various Temps - %s" % get_name_of_experiment(output_fn_base), xlab="Initial Temperature")

    plot_multiple([fitness_scores_geo],
                   ["Geometric Decay Fitness"],
                   "%s/Annealing/sa_fitness_decay_geo" % output_fn_base,
                   title="SA Fitness Over Various Temps - Geometric Decay - %s" % get_name_of_experiment(output_fn_base), xlab="Initial Temperature")
    plot_multiple([fitness_scores_arith],
                   ["Arithmetic Decay Fitness"],
                   "%s/Annealing/sa_fitness_decay_arith" % output_fn_base,
                   title="SA Fitness Over Various Temps - Arithmetic Decay - %s" % get_name_of_experiment(output_fn_base), xlab="Initial Temperature")
    plot_multiple([fitness_scores_exp],
                   ["Exponential Decay Fitness"],
                   "%s/Annealing/sa_fitness_decay_exp" % output_fn_base,
                   title="SA Fitness Over Various Temps - Exponential Decay - %s" % get_name_of_experiment(output_fn_base), xlab="Initial Temperature")

    plot_multiple([pd.Series(fitness_scores_geo).rolling(window=10).mean(), pd.Series(fitness_scores_arith).rolling(window=10).mean(), pd.Series(fitness_scores_exp).rolling(window=10).mean()],
                   ["Geometric Decay Fitness", "Arithmetic Decay Fitness", "Exponential Decay Fitness"],
                   "%s/Annealing/sa_fitness_decay_sma" % output_fn_base,
                   title="SA Fitness Scores Over Various Temps - Rolling Mean - %s" % get_name_of_experiment(output_fn_base), xlab="Initial Temperature")

    logs.append("\tHyperparameters: \n")
    logs.append("\t%s" % str(hparams))
    logs.append("\n\n\tBest State: \n\t\t")
    logs.append(str(list(best_state)))
    logs.append("\n\tBest Fitness: \n\t\t")
    logs.append(str(best_fitness))


    return logs, metrics

def get_name_of_experiment(fn):
    if(fn == "one_max"):
        return "One Max"
    elif(fn == "four_peaks"):
        return "Four Peaks"
    elif(fn == "tsp"):
        return "TSP"
    else:
        return fn


def genetic_experiment(optimization_problem, hparams, output_fn_base):
    metrics = {}
    logs = []
    print("----Running Genetic Algorithm Experiment-----")
    print("Hyperparameters Used: ")
    print(hparams)
    max_iters_for_test = 150
    best_state, best_fitness, _ = mlrose.genetic_alg(optimization_problem,pop_size=hparams["pop_size"], mutation_prob=hparams["mutation_prob"], max_attempts= hparams["max_attempts"], max_iters=hparams["max_iters"], pop_breed_percent=.2)

    # Iterations and runtimes
    fitness_scores = []
    runtimes = []
    iteration_count = range(1,hparams["max_iters"])
    for iter in iteration_count:
        start_time = time.time()
        best_state, best_fitness, _ = mlrose.genetic_alg(optimization_problem,pop_size=hparams["pop_size"], mutation_prob=hparams["mutation_prob"], max_attempts= hparams["max_attempts"], max_iters= iter, pop_breed_percent=.2)
        end_time = time.time()
        runtimes.append(end_time - start_time)
        fitness_scores.append(-best_fitness if optimization_problem.prob_type == 'tsp' else best_fitness)

    plot_single(fitness_scores,"%s/Genetic/genetic_fitness_iterations" % output_fn_base, xvals=iteration_count)
    plot_runtimes(runtimes,"%s/Genetic/genetic_runtime_iterations" % output_fn_base)

    metrics["runtimes"] = runtimes
    metrics["fitness"] = fitness_scores

    # Population
    fitness_scores = []
    population_count = range(10,hparams["pop_size"])
    for pop in population_count:
        best_state, best_fitness, _ = mlrose.genetic_alg(optimization_problem,pop_size=pop, mutation_prob=hparams["mutation_prob"], max_attempts= hparams["max_attempts"], max_iters=hparams["max_iters"], pop_breed_percent=.2)
        fitness_scores.append(-best_fitness if optimization_problem.prob_type == 'tsp' else best_fitness)

    plot_single(fitness_scores,"%s/Genetic/genetic_fitness_population" % output_fn_base,title="Genetic Algorithm Fitness As Population Grows - %s" % get_name_of_experiment(output_fn_base), xlab="Population Size",xvals=population_count)

    # Mutation Probability
    fitness_scores = []
    probs = np.arange(0.1, 1.0, 0.1)
    for prob in probs:
        best_state, best_fitness, _ = mlrose.genetic_alg(optimization_problem,pop_size=hparams["pop_size"], mutation_prob=prob, max_attempts= hparams["max_attempts"], max_iters=max_iters_for_test, pop_breed_percent=.2,)
        fitness_scores.append(-best_fitness if optimization_problem.prob_type == 'tsp' else best_fitness)

    plot_single(fitness_scores,"%s/Genetic/genetic_fitness_mutation_prob" % output_fn_base,title="Genetic Algorithm Fitness As Mutation Prob. Grows - %s" % get_name_of_experiment(output_fn_base), xlab="Mutation Probability", xvals=probs)

    # pop_breed_percent
    fitness_scores = []
    pops = np.arange(0.2, 1.0, 0.1)
    for pop in pops:
        best_state, best_fitness, _ = mlrose.genetic_alg(optimization_problem,pop_size=hparams["pop_size"], mutation_prob=hparams["mutation_prob"], max_attempts= hparams["max_attempts"], max_iters=hparams["max_iters"], pop_breed_percent=pop)
        fitness_scores.append(-best_fitness if optimization_problem.prob_type == 'tsp' else best_fitness)

    plot_single(fitness_scores,"%s/Genetic/genetic_fitness_breed_percent" % output_fn_base,title="Genetic Algorithm Fitness As Population Breed Percentage Grows - %s" % get_name_of_experiment(output_fn_base), xlab="Breed Percentage", xvals=pops)

    logs.append("\tHyperparameters: \n")
    logs.append("\t%s" % str(hparams))
    logs.append("\n\n\tBest State: \n\t\t")
    logs.append(str(list(best_state)))
    logs.append("\n\tBest Fitness: \n\t\t")
    logs.append(str(best_fitness))


    return logs, metrics

def mimic_experiment(optimization_problem, hparams, output_fn_base):
    logs = []
    metrics = {}
    print("----Running MIMIC Experiment-----")
    print("Hyperparameters Used: ")
    print(hparams)

    best_state, best_fitness, _ = mlrose.mimic(optimization_problem, pop_size=hparams["pop_size"],keep_pct=hparams["keep_pct"], max_attempts= hparams["max_attempts"], max_iters= hparams["max_iters"])

    # Iterations
    fitness_scores = []
    runtimes = []
    iteration_count = range(1,hparams["max_iters"])
    for iter in iteration_count:
        start_time = time.time()
        best_state, best_fitness, _ = mlrose.mimic(optimization_problem, pop_size=hparams["pop_size"],keep_pct=hparams["keep_pct"], max_attempts=hparams["max_attempts"], max_iters=iter)
        end_time = time.time()
        runtimes.append(end_time - start_time)
        fitness_scores.append(-best_fitness if optimization_problem.prob_type == 'tsp' else best_fitness)

    plot_single(fitness_scores,"%s/MIMIC/mimic_fitness_iterations" % output_fn_base, xvals=iteration_count)
    plot_runtimes(runtimes,"%s/MIMIC/mimic_runtime_iterations" % output_fn_base)
    metrics["runtimes"] = runtimes
    metrics["fitness"] = fitness_scores

    # Population
    fitness_scores = []
    population_count = range(1,hparams["pop_size"])
    for pop in population_count:
        best_state, best_fitness, _ = mlrose.mimic(optimization_problem,pop_size=pop,keep_pct=hparams["keep_pct"],max_attempts= hparams["max_attempts"],max_iters= hparams["max_iters"])
        fitness_scores.append(-best_fitness if optimization_problem.prob_type == 'tsp' else best_fitness)

    plot_single(fitness_scores,"%s/MIMIC/mimic_fitness_population" % output_fn_base,title="MIMIC Fitness As Population Grows - %s" % get_name_of_experiment(output_fn_base), xlab="Population Size", xvals=population_count)

    logs.append("\tHyperparameters: \n")
    logs.append("\t%s" % str(hparams))
    logs.append("\n\n\tBest State: \n\t\t")
    logs.append(str(list(best_state)))
    logs.append("\n\tBest Fitness: \n\t\t")
    logs.append(str(best_fitness))


    return logs, metrics

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

    return train_sizes, train_scores, test_scores, fit_times

def create_confusion_matrix(clf, X, y, cv, file_name):
    y_train_pred = cross_val_predict(clf, X, y, cv=cv)
    conf_mx = confusion_matrix(y,y_train_pred)
    plt.matshow(conf_mx,cmap=plt.cm.gray)
    plt.savefig(file_name)
    plt.close()
    return conf_mx

# https://stackoverflow.com/questions/44478133/how-to-set-initial-weights-in-mlpclassifier
class MLPClassifierOverride(MLPClassifier):

    def set_weights(self, coefs, intercepts):
        self.coefs = coefs
        self.intercepts = intercepts

    def _init_coef(self, fan_in, fan_out, dtype):

        coef_init = self.coefs

        intercept_init = self.intercepts

        return coef_init, intercept_init

class MLPClassifierFitness:

    def __init__(self, mlp_clf, X_test, y_test, X, y, cross_v=10):
        self.clf = mlp_clf
        self.X_test = X_test
        self.y_test = y_test
        self.cv = cross_v
        self.X = X
        self.y = y
        self.prob_type = 'continuous'


    def evaluate(self, state):

        self.clf.set_weights()

        # classification report:
        report = classification_report(self.y_test, self.clf.predict(self.X_test))

        # cross validation score
        cvs = cross_val_score(self.clf, self.X, self.y, cv=10, scoring='accuracy')


        # Evaluate function
        fitness = 0 #max(max_0, max_1) + _r

        return fitness


    def get_prob_type(self):
        """ Return the problem type.

        Returns
        -------
        self.prob_type: string
            Specifies problem type as 'discrete', 'continuous', 'tsp'
            or 'either'.
        """
        return self.prob_type


def neural_network_experiment(dataset, hparams, output_fn_base):

    logs = []
    metrics_dictionary={}


    print("----Running Neural Network Experiment-----")
    X = dataset["features"]
    y = dataset["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=True,test_size=.2, train_size=.8)
    # scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # One hot encode target values
    y_test = y_test.to_numpy()
    y_train = y_train.to_numpy()
    one_hot = OneHotEncoder()

    y_train = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    y_test = one_hot.transform(y_test.reshape(-1, 1)).todense()

    # hill climb ################################################################################
    logs.append("\nHill climb + NN ###########################################\n")
    nn_model2 = mlrose.NeuralNetwork(hidden_nodes=[50], activation='tanh',
                                     algorithm='random_hill_climb', max_iters=250,
                                     bias=True, is_classifier=True, learning_rate=0.001,
                                     early_stopping=True, clip_max=5, max_attempts=100, curve=True)
    nn_model2.fit(X_train, y_train)
    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model2.predict(X_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    train_sizes, train_scores, test_scores, fit_times = plot_learning_curve2(nn_model2, "random_hill_curve", "Learning Curve - Random Hill Climb + Neural Network", X, y)
    # classification report:
    logs.append(classification_report(y_test, nn_model2.predict(X_test)))

    # confusion matrix
    # matrix = create_confusion_matrix(nn_model2, X, y, 10, "./Confusion_hill_%s" % output_fn_base )
    # cross validation score
    cvs = cross_val_score(nn_model2, X, y, cv=10, scoring='accuracy')
    logs.append("\n\n\tConfusion Matrix: \n")
    # logs.append("\t%s\n" % str(matrix).replace("\n", "\n\t"))
    logs.append("\n\tMean 10-fold CV Score %.02f\n" % cvs.mean())
    logs.append("\tTest Accuracy: %.02f\n" % y_test_accuracy)

    metrics_dictionary["validation_scores_hill"] = test_scores
    metrics_dictionary["fit_times_hill"] = fit_times
    metrics_dictionary["train_sizes"] = train_sizes
    # simulated_annealing ################################################################################
    logs.append("\nSimulated Annealing + NN ###########################################\n")
    nn_model2 = mlrose.NeuralNetwork(hidden_nodes=[50], activation='tanh',
                                     algorithm='simulated_annealing', max_iters=250,
                                     bias=True, is_classifier=True, learning_rate=0.001,
                                     early_stopping=True, clip_max=5, max_attempts=100, curve=True)
    nn_model2.set_params()
    nn_model2.fit(X_train, y_train)
    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model2.predict(X_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    train_sizes, train_scores, test_scores, fit_times = plot_learning_curve2(nn_model2, "simulated_annealing", "Learning Curve - Simulated Annealing + Neural Network", X, y)
    # classification report:
    logs.append(classification_report(y_test, nn_model2.predict(X_test)))

    # confusion matrix
    # matrix = create_confusion_matrix(nn_model2, X, y, 10, "./Confusion_sa_%s" % output_fn_base )
    # cross validation score
    cvs = cross_val_score(nn_model2, X, y, cv=10, scoring='accuracy')
    logs.append("\n\n\tConfusion Matrix: \n")
    # logs.append("\t%s\n" % str(matrix).replace("\n", "\n\t"))
    logs.append("\n\tMean 10-fold CV Score %.02f\n" % cvs.mean())
    logs.append("\tTest Accuracy: %.02f\n" % y_test_accuracy)
    metrics_dictionary["validation_scores_sa"] = test_scores
    metrics_dictionary["fit_times_sa"] = fit_times

    # genetic algorithm ################################################################################
    logs.append("\nGenetic Algorithm + NN ###########################################\n")
    nn_model2 = mlrose.NeuralNetwork(hidden_nodes=[50], activation='tanh',
                                     algorithm='genetic_alg', max_iters=250,
                                     bias=True, is_classifier=True, learning_rate=0.001,
                                     early_stopping=True, clip_max=5, max_attempts=100, curve=True)
    nn_model2.fit(X_train, y_train)
    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model2.predict(X_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    train_sizes, train_scores, test_scores, fit_times = plot_learning_curve2(nn_model2, "genetic_alg", "Learning Curve - Genetic Algorithm + Neural Network", X, y)
    # classification report:
    logs.append(classification_report(y_test, nn_model2.predict(X_test)))

    # confusion matrix
    # matrix = create_confusion_matrix(nn_model2, X, y, 10, "./Confusion_genetic_%s" % output_fn_base )
    # cross validation score
    cvs = cross_val_score(nn_model2, X, y, cv=10, scoring='accuracy')
    logs.append("\n\n\tConfusion Matrix: \n")
    # logs.append("\t%s\n" % str(matrix).replace("\n", "\n\t"))
    logs.append("\n\tMean 10-fold CV Score %.02f\n" % cvs.mean())
    logs.append("\tTest Accuracy: %.02f\n" % y_test_accuracy)
    metrics_dictionary["validation_scores_genetic"] = test_scores
    metrics_dictionary["fit_times_genetic"] = fit_times



    # # from paper 1
    logs.append("\nAssignment 1 NN ###########################################\n")
    nn_clf = MLPClassifier(solver=hparams["solver"], alpha=hparams["alpha"],hidden_layer_sizes=hparams["hidden_layers"], max_iter=hparams["epochs"], activation=hparams["activation"], early_stopping=True, learning_rate=hparams["learning_rate"], learning_rate_init= hparams["learning_rate_init"])

    # learning curve
    cv = ShuffleSplit(n_splits=100, test_size=0.2)
    train_scores, train_sizes, validation_scores, fit_times = plot_learning_curve2(nn_clf, "original_mlp", "Learning Curve - Neural Network (Gradient Descent Baseline)", X, y, ylim=None, cv=cv, n_jobs=1)
    metrics_dictionary["validation_scores_gradient"] = validation_scores
    metrics_dictionary["fit_times_gradient"] = fit_times

    nn_clf.fit(X_train, y_train)

    # classification report:
    logs.append(classification_report(y_test, nn_clf.predict(X_test)))

    # confusion matrix
    # matrix = create_confusion_matrix(nn_clf, X, y, 10, "%s/NN/Confusion_%s" % (output_fn_base, output_fn_base))

    # cross validation score
    cvs = cross_val_score(nn_clf, X, y, cv=10, scoring='accuracy')
    print("Mean 10-fold CV Score: %.02f" % cvs.mean())




    return logs, metrics_dictionary

if __name__ == "__main__":
    print("#####################################################################")
    print("CS7641 ML - Randomized Optimization Assignment Test Program")
    print("#####################################################################")
    optimization_probs = get_fitness_functions()
    if not os.path.exists('./aggregated_runtime_results'):
        os.makedirs('./aggregated_runtime_results')
    if not os.path.exists('./aggregated_fitness_results'):
        os.makedirs('./aggregated_fitness_results')
    # Writing to file
    with open("metrics.txt", "w") as file1:
        file1.write("CS7641 ML - Randomized Optimization Assignment Test Program \n")
        file1.write("Derek Chase Brown (dbrown381@gatech.edu) \n")
        file1.write("\n============ Metrics ============ \n\n")
        # Test workbenches:
        for problem_name in optimization_probs:
            file1.write("DATASET: %s\n---------------------------------------------------------\n\n" % problem_name)
            print("Testing on dataset: %s" % problem_name )
            if not os.path.exists('%s/Hill' % problem_name):
                os.makedirs('%s/Hill' % problem_name)
            if not os.path.exists('%s/Annealing' % problem_name):
                os.makedirs('%s/Annealing' % problem_name)
            if not os.path.exists('%s/Genetic' % problem_name):
                os.makedirs('%s/Genetic' % problem_name)
            if not os.path.exists('%s/MIMIC' % problem_name):
                os.makedirs('%s/MIMIC' % problem_name)


            hc_logs, hc_metrics = hill_climb_experiment(optimization_probs[problem_name],get_hyperparams(problem_name,"Hill"), problem_name)
            file1.write("Hill Climb Metrics ------------- \n")
            file1.writelines(hc_logs)

            sa_logs, sa_metrics = simulated_annealing_experiment(optimization_probs[problem_name],get_hyperparams(problem_name,"SA"), problem_name)
            file1.write("Simulated Annealing Metrics ------------- \n")
            file1.writelines(sa_logs)

            ga_logs, ga_metrics = genetic_experiment(optimization_probs[problem_name],get_hyperparams(problem_name,"GA"), problem_name)
            file1.write("Genetic Algorithm Metrics ------------- \n")
            file1.writelines(ga_logs)

            mimic_logs, mimic_metrics = mimic_experiment(optimization_probs[problem_name],get_hyperparams(problem_name,"MIMIC"), problem_name)
            file1.write("MIMIC Metrics ------------- \n")
            file1.writelines(mimic_logs)

            plot_multiple([hc_metrics["fitness"], sa_metrics["fitness"], ga_metrics["fitness"], mimic_metrics["fitness"]],
                           ["Hill Climb", "Simulated Annealing", "Genetic Algorithm", "MIMIC"],
                           "aggregated_fitness_results/%s_fitness" % problem_name,
                           title="Aggregated Performance VS Iterations - %s" % problem_name, xlab="Number of Iterations")

            plot_multiple([hc_metrics["runtimes"], sa_metrics["runtimes"], ga_metrics["runtimes"], mimic_metrics["runtimes"]],
                           ["Hill Climb", "Simulated Annealing", "Genetic Algorithm", "MIMIC"],
                           "aggregated_runtime_results/%s_runtime" % problem_name,
                           title="Aggregated Runtimes VS Iterations - %s" % problem_name, xlab="Number of Iterations", ylab="Runtimes")

            print()
        hparams = {
                    # "first_hidden_layer_node_count": 9,
                    # "hidden_layers": [{"count": 100, "activation": "relu"}],
                    # "loss_fn": 'sparse_categorical_crossentropy',
                    "epochs": 200,
                    "solver": "adam",
                    "activation": "tanh",
                    "hidden_layers": (50,),
                    "learning_rate": 'constant',
                    "alpha": .5, #1e-5
                    "learning_rate_init": .01
                }
        # set_A = read_csv("./cardiotocogram.csv")
        # dataset = {"features": set_A.drop(["Class"], axis=1), "class": set_A["Class"]-1}

        set_A = read_csv("./shill.csv")
        set_A = set_A.drop(['Record_ID', 'Auction_ID', 'Bidder_ID'], axis=1)
        set_A.dropna(inplace=True)
        dataset = {"features": set_A.drop(["Class"], axis=1), "class": set_A["Class"]}
        logs, metrics = neural_network_experiment(dataset=dataset, hparams=hparams,output_fn_base="Shill")

        plot_multiple([np.mean(metrics["fit_times_genetic"],axis=1), np.mean(metrics["fit_times_sa"],axis=1), np.mean(metrics["fit_times_hill"],axis=1), np.mean(metrics["fit_times_gradient"],axis=1)],
                       ["Genetic Algorithm", "Simulated Annealing", "Randomized Hill Climb", "Gradient Descent"],
                       "nn_%s_runtimes" % "Shill",
                       title="Aggregated Runtimes - NN + Optimization Algorithms - %s" % "Shill", xlab="Cross Validation Accuracy", ylab="Runtimes",xvals=metrics["train_sizes"])

        plot_multiple([np.mean(metrics["validation_scores_genetic"],axis=1), np.mean(metrics["validation_scores_sa"],axis=1), np.mean(metrics["validation_scores_hill"],axis=1), np.mean(metrics["validation_scores_gradient"],axis=1)],
                       ["Genetic Algorithm", "Simulated Annealing", "Randomized Hill Climb", "Gradient Descent"],
                       "nn_%s_performance" % "Shill",
                       title="Aggregated Performance - NN + Optimization Algorithms - %s" % "Shill", xlab="Cross Validation Accuracy", ylab="Score", xvals=metrics["train_sizes"])

        file1.writelines(logs)

