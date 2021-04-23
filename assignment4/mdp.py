import numpy as np
import time
from pandas import read_csv
import pandas as pd
import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp
import mdptoolbox.mdp
import matplotlib
import matplotlib.pyplot as plt
import gym
import seaborn as sns
from threading import Thread
import concurrent.futures

import os

matplotlib.use("TkAgg")

HYPERPARAMS = {

}

def plot_single(x, y, fn, title, xlab="Iterations Taken", ylab="Time", convergenceX = -1):
    # Draw lines
    plt.plot(x,y, linewidth=2, label=ylab)

    # draw convergence
    if(convergenceX >= 0):
        plt.vlines(x=convergenceX, ymin=np.min(y), ymax=np.max(y), colors='teal', lw=2, label='Convergence Point (X=%d)' % convergenceX, linestyles="--")
        #plt.axvline(x=convergenceX)

    # Create plot
    plt.title(title)
    plt.xlabel(xlab), plt.ylabel(ylab)
    plt.tight_layout()
    plt.legend()
    plt.savefig("./%s" % fn)
    plt.close()


"""
Value Iteration
- Convergence:
    According to [Norvig, Russell, AI Modern Approach], value iteration convergences when an 'equilibrium', which happens
    after running through several iterations of Bellman updates - updating utilities for a given state from its neighbors'
    utilities. Utilities are initialized as arbitrary. Because there are n states and thus n bellman equations, VI converges
    eventually to a unique set of solutions. It turns out the algorithm always converges as long as gamma is less than 1.
    It actually converges at an exponential rate, since error is reduced by a factor of gamma.

- Experimentation
    Both the discount factor and stopping factor gamma are provided by mdptoolbox.

- Problems
    If an agent only cares about the policy (as we would expect), value iteration is a bit 'greedy' or 'extreme' when
    compared to policy iteration. By the time value iteration converges, the actual policy itself may have converged 
    much earlier. Because of this, value iteration may be slower than PI.

- Other Notes
    The algorithm cares about the performance of itself based on decisions made through a utility function

    Worst case - O(M*N^2)[https://arxiv.org/ftp/arxiv/papers/1302/1302.4971.pdf], where M represents actions, N represents
    states.


"""
def value_iteration_experiment(P, R, problem_name, max_iter=750):
    convergences = {
        "Taxi": {
            "Best": {
                "gamma" : .98,
                "epsilon": .1,
            }

        },
        "ForestSmall": {
            "Best": {
                "gamma" : .98,
                "epsilon": .05,
            }
        },
        "ForestMedium": {
            "Best": {
                "gamma" : .98,
                "epsilon": .05,
            }
        },
        "ForestLarge": {
            "Best": {
                "gamma" : .92,
                "epsilon": .05,
            }
        }
    }
    if not os.path.exists('./%s/value_iteration' % problem_name):
        os.makedirs('./%s/value_iteration' % problem_name)
    returned_metrics = {}
    """
    # First Run - Solve each MDP using value iteration as well as policy iteration. How many iterations does it take to converge?
    In the first run, run an intitial experiment to check iterations for convergence and runtime
    """
    if not os.path.exists('./%s/value_iteration/first_run' % problem_name):
        os.makedirs('./%s/value_iteration/first_run' % problem_name)

    vi = hiive.mdptoolbox.mdp.ValueIteration(P,R,gamma=convergences[problem_name]["Best"]["gamma"],epsilon=convergences[problem_name]["Best"]["epsilon"],max_iter=max_iter)
    vi.setVerbose()
    vi.max_iter = max_iter
    data = vi.run()
    returned_policy = vi.policy
    states, actions, rewards, error, time, maxV, meanV, iteration = policy_value_iteration_stats_to_tuple(data)
    returned_reward = np.max(rewards)
    error_converge = len(error)

    plot_single(x=iteration,y=time,title="Value Iteration - %s - Time Over Iteration" % problem_name,
                fn = "%s/value_iteration/first_run/iteration_time" % (problem_name),xlab="Iteration",ylab= "Time (s)",
                convergenceX=error_converge)

    plot_single(x=iteration,y=error,title="Value Iteration - %s - Error Over Iteration" % problem_name,
                fn = "%s/value_iteration/first_run/iteration_error" % (problem_name),xlab="Iteration",ylab= "Error",
                convergenceX=error_converge)

    plot_single(x=iteration,y=rewards,title="Value Iteration - %s - Reward Over Iteration" % problem_name,
                fn = "%s/value_iteration/first_run/iteration_reward" % (problem_name),xlab="Iteration",ylab= "Reward")

    plot_single(x=iteration,y=maxV,title="Value Iteration - %s - maxV Over Iteration" % problem_name,
                fn = "%s/value_iteration/first_run/iteration_v_max" % (problem_name),xlab="Iteration",ylab= "V_Max")

    plot_single(x=iteration,y=meanV,title="Value Iteration - %s - meanV Over Iteration" % problem_name,
                fn = "%s/value_iteration/first_run/iteration_v_mean" % (problem_name),xlab="Iteration",ylab= "V_Mean")

    """
    # Second Run - Vary Gamma
    """
    if not os.path.exists('./%s/value_iteration/second_run' % problem_name):
        os.makedirs('./%s/value_iteration/second_run' % problem_name)

    xs = []
    convergence_points = []
    convergence_rewards = []
    convergence_runtimes = []
    for i in range(1, 100, 2):

        try:
            vi = hiive.mdptoolbox.mdp.ValueIteration(P,R,gamma=i/100,epsilon=.9,max_iter=max_iter)
            # pi.epsilon
            # pi.setVerbose()
            vi.max_iter = max_iter
            data = vi.run()


            states, actions, rewards, error, time, maxV, meanV, iteration = policy_value_iteration_stats_to_tuple(data)
            converging_iter = len(error)
            xs.append(i / 100)
            convergence_points.append(converging_iter)
            convergence_rewards.append(rewards[converging_iter-1])
            convergence_runtimes.append(time[converging_iter-1])
        except:
            continue # case where no convergence


    plot_single(x=xs, y=convergence_points,
                title="Value Iteration - %s - Iterations To Converge As Gamma Changes" % problem_name,
                fn="%s/value_iteration/second_run/gamma_v_convergence" % (problem_name), xlab="Gamma",
                ylab="Convergence Iteration")

    plot_single(x=xs, y=convergence_runtimes,
                title="Value Iteration - %s - Runtime As Gamma Changes" % problem_name,
                fn="%s/value_iteration/second_run/gamma_v_runtime" % (problem_name), xlab="Gamma",
                ylab="Runtime (s)")

    plot_single(x=xs, y=convergence_rewards,
                title="Value Iteration - %s - Iterations To Converge As Gamma Changes" % problem_name,
                fn="%s/value_iteration/second_run/gamma_v_reward" % (problem_name), xlab="Gamma",
                ylab="Converging Reward")
    returned_metrics["second_run"] = {
        "xs": xs,
        "convergence_runtimes": convergence_runtimes,
        "convergence_rewards": convergence_rewards,
        "convergence_points": convergence_points
    }

    """
    # Third Run - Vary Epsilon
    """
    if not os.path.exists('./%s/value_iteration/third_run' % problem_name):
        os.makedirs('./%s/value_iteration/third_run' % problem_name)

    xs = []
    convergence_points = []
    convergence_rewards = []
    convergence_runtimes = []
    for i in range(1, 100, 2):

        try:
            pi = hiive.mdptoolbox.mdp.ValueIteration(P,R,gamma=.98,epsilon=i/100,max_iter=max_iter)
            # pi.eval_type="iterative"
            pi.epsilon = i / 100
            # pi.setVerbose()
            pi.max_iter = max_iter
            data = pi.run()

            states, actions, rewards, error, time, maxV, meanV, iteration = policy_value_iteration_stats_to_tuple(data)
            converging_iter = len(error)
            xs.append(i / 100)
            convergence_points.append(converging_iter)
            convergence_rewards.append(rewards[converging_iter-1])
            convergence_runtimes.append(time[converging_iter-1])
        except:
            continue
    plot_single(x=xs, y=convergence_points,
                title="Value Iteration - %s - Iterations To Converge As Epsilon Changes" % problem_name,
                fn="%s/value_iteration/third_run/epsilon_v_convergence" % (problem_name), xlab="Epsilon",
                ylab="Convergence Iteration")

    plot_single(x=xs, y=convergence_runtimes,
                title="Value Iteration - %s - Runtime As Gamma Changes" % problem_name,
                fn="%s/value_iteration/third_run/gamma_v_runtime" % (problem_name), xlab="Gamma",
                ylab="Runtime (s)")

    plot_single(x=xs, y=convergence_rewards,
                title="Value Iteration - %s - Iterations To Converge As Gamma Changes" % problem_name,
                fn="%s/value_iteration/third_run/gamma_v_reward" % (problem_name), xlab="Gamma",
                ylab="Converging Reward")

    returned_metrics["third_run"] = {
        "xs": xs,
        "convergence_runtimes": convergence_runtimes,
        "convergence_rewards": convergence_rewards,
        "convergence_points": convergence_points
    }


    print("Value Iteration Run Complete")
    return returned_policy, returned_reward, returned_metrics

"""
Policy Iteration
- Convergence:
    According to [Norvig, Russell, AI Modern Approach], policy iteration convergence is straightforward in that it is
    converged simply at the moment that an iteration results in 0 change in utility. There are a finite number of policies 
    in a finite state space, thus policy iteration must terminate.

- Experimentation
    The algorithm is modified in mdptoolbox to allow for not only the discount factor, but also an epsilon. Convergence
    is based off variation. Both the discount factor and the epsilon value are used to determine the threshold needed
    to indicate that variation has dropped below acceptable levels and allows the algorithm to be called converged.
    According to [Norvig, Russell, AI Modern Approach], the discount factor gamma is to be set closer to 0 when it is 
    desired that rewards in the distant future is to be viewed as insignificant. Else it should be closer to 1. When equal
    to 1, the discounted rewards are exactly equivalent to additive rewards. 

- Problems


- Other Notes
    Using the notes on convergence and experimentation, it is important to note that MDPToolBox's implementation of policy
    iteration approaches the problem of variation and convergence by declaring that convergence happens when variance is
    less than ((1 - gamma) / gamma) * epsilon.
    
    Policy iteration is based off the idea that utility values on states don't have to be precise if on action is better
    than all the others by an obvious observation. [Norvig, Russell, AI Modern Approach]
    
    Worst case - O(n^3)[Norvig, Russell, AI Modern Approach]


"""
def policy_iteration_experiment(P, R, problem_name, max_iter=750):
    convergences = {
        "Taxi": {
            "Best": {
                "gamma" : .98,
            }

        },
        "ForestSmall": {
            "Best": {
                "gamma" : .98,
            }
        },
        "ForestMedium": {
            "Best": {
                "gamma" : .98,
            }
        },
        "ForestLarge": {
            "Best": {
                "gamma" : .98,
            }
        }
    }
    returned_policy = None
    returned_metrics = {}
    if not os.path.exists('./%s/policy_iteration' % problem_name):
        os.makedirs('./%s/policy_iteration' % problem_name)

    """
    # First Run - Solve each MDP using value iteration as well as policy iteration. How many iterations does it take to converge?
    In the first run, we determine what a totally random e-greedy approach is like. This experiment is the highest level of 
    exploration. In this, it can likely be noticed that 
    """
    if not os.path.exists('./%s/policy_iteration/first_run' % problem_name):
        os.makedirs('./%s/policy_iteration/first_run' % problem_name)
    pi = hiive.mdptoolbox.mdp.PolicyIteration(P, R,gamma=convergences[problem_name]["Best"]["gamma"]
                                              ,max_iter=max_iter)
    #pi.epsilon
    pi.setVerbose()
    pi.max_iter = max_iter
    data = pi.run()
    returned_policy = pi.policy

    states, actions, rewards, error, time, maxV, meanV, iteration = policy_value_iteration_stats_to_tuple(data)

    error_converge = thresh_convergence_test(np.array(error),.002, 1)
    returned_reward = np.max(rewards)

    plot_single(x=iteration,y=time,title="Policy Iteration - %s - Time Over Iteration" % problem_name,
                fn = "%s/policy_iteration/first_run/iteration_time" % (problem_name),xlab="Iteration",ylab= "Time (s)",
                convergenceX=error_converge)

    plot_single(x=iteration,y=error,title="Policy Iteration - %s - Error Over Iteration" % problem_name,
                fn = "%s/policy_iteration/first_run/iteration_error" % (problem_name),xlab="Iteration",ylab= "Error",
                convergenceX=error_converge)

    plot_single(x=iteration,y=rewards,title="Policy Iteration - %s - Reward Over Iteration" % problem_name,
                fn = "%s/policy_iteration/first_run/iteration_reward" % (problem_name),xlab="Iteration",ylab= "Reward")

    plot_single(x=iteration,y=maxV,title="Policy Iteration - %s - maxV Over Iteration" % problem_name,
                fn = "%s/policy_iteration/first_run/iteration_v_max" % (problem_name),xlab="Iteration",ylab= "V_Max")

    plot_single(x=iteration,y=meanV,title="Policy Iteration - %s - meanV Over Iteration" % problem_name,
                fn = "%s/policy_iteration/first_run/iteration_v_mean" % (problem_name),xlab="Iteration",ylab= "V_Mean")

    """
    # Second Run - Vary Gamma
    In the first run, we determine what a totally random e-greedy approach is like. This experiment is the highest level of
    exploration. In this, it can likely be noticed that
    """
    if not os.path.exists('./%s/policy_iteration/second_run' % problem_name):
        os.makedirs('./%s/policy_iteration/second_run' % problem_name)


    xs = []
    convergence_points = []
    convergence_runtimes = []
    convergence_rewards = []
    for i in range(1,100):

        try:
            pi = hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma=i/100, max_iter=max_iter, eval_type=1)
            # pi.epsilon
            #pi.setVerbose()
            pi.max_iter = max_iter
            data = pi.run()
            xs.append(i / 100)
            states, actions, rewards, error, time, maxV, meanV, iteration = policy_value_iteration_stats_to_tuple(data)
            converging_iter = len(error)
            convergence_points.append(converging_iter)
            convergence_rewards.append(rewards[converging_iter-1])
            convergence_runtimes.append(time[converging_iter-1])
        except:
            continue

    plot_single(x=xs, y=convergence_points,
                title="Policy Iteration - %s - Iterations To Converge As Gamma Changes" % problem_name,
                fn="%s/policy_iteration/second_run/gamma_v_convergence" % (problem_name), xlab="Gamma",
                ylab="Convergence Iteration")

    plot_single(x=xs, y=convergence_runtimes,
                title="Policy Iteration - %s - Runtime As Gamma Changes" % problem_name,
                fn="%s/policy_iteration/second_run/gamma_v_runtime" % (problem_name), xlab="Gamma",
                ylab="Runtime (s)")

    plot_single(x=xs, y=convergence_rewards,
                title="Policy Iteration - %s - Iterations To Converge As Gamma Changes" % problem_name,
                fn="%s/policy_iteration/second_run/gamma_v_reward" % (problem_name), xlab="Gamma",
                ylab="Converging Reward")

    returned_metrics["second_run"] = {
        "xs": xs,
        "convergence_runtimes": convergence_runtimes,
        "convergence_rewards": convergence_rewards,
        "convergence_points": convergence_points
    }

    """
    # Third Run - Vary Epsilon
    In the first run, we determine what a totally random e-greedy approach is like. This experiment is the highest level of
    exploration. In this, it can likely be noticed that
    """
    if not os.path.exists('./%s/policy_iteration/third_run' % problem_name):
        os.makedirs('./%s/policy_iteration/third_run' % problem_name)


    xs = []
    convergence_points = []
    convergence_runtimes = []
    convergence_rewards = []
    for i in range(1,10):

        try:
            pi = hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma=.98, max_iter=max_iter, eval_type=1)
            #pi.eval_type="iterative"
            pi.epsilon = i/10
            #pi.setVerbose()
            pi.max_iter = max_iter
            data = pi.run()
            xs.append(i / 10)
            states, actions, rewards, error, time, maxV, meanV, iteration = policy_value_iteration_stats_to_tuple(data)
            converging_iter = len(error)
            convergence_points.append(converging_iter)
            convergence_rewards.append(rewards[converging_iter-1])
            convergence_runtimes.append(time[converging_iter-1])
        except:
            continue

    plot_single(x=xs, y=convergence_points,
                title="Policy Iteration - %s - Iterations To Converge As Epsilon Changes" % problem_name,
                fn="%s/policy_iteration/third_run/epsilon_v_convergence" % (problem_name), xlab="Epsilon",
                ylab="Convergence Iteration")

    plot_single(x=xs, y=convergence_runtimes,
                title="Policy Iteration - %s - Runtime As Gamma Changes" % problem_name,
                fn="%s/policy_iteration/third_run/epsilon_v_runtime" % (problem_name), xlab="Gamma",
                ylab="Runtime (s)")

    plot_single(x=xs, y=convergence_rewards,
                title="Policy Iteration - %s - Iterations To Converge As Gamma Changes" % problem_name,
                fn="%s/policy_iteration/third_run/epsilon_v_reward" % (problem_name), xlab="Gamma",
                ylab="Converging Reward")

    returned_metrics["third_run"] = {
        "xs": xs,
        "convergence_runtimes": convergence_runtimes,
        "convergence_rewards": convergence_rewards,
        "convergence_points": convergence_points
    }

    print("Policy Iteration Run Complete")
    return returned_policy, returned_reward, returned_metrics

"""
Q Learning
- Convergence:
    According to [Mitchell, Machine Learning], Q learning is guaranteed to converge under certain conditions:
        1. MDP is deterministic
        2. Immediate Reward Values are bounded
        3. Assume the agent selects actions in such a way that it visits every possible state-action pair infinitely often
    But the key to why it works is because the table entry with the largest error must have its error reduced by a factor
    of gamma when it is updated.
    Q learning's convergence theorem implies that Q learning need not train on optimal action sequences in order to converge.
    
    

- Experimentation
    According to [Geron, Hands on Machine Learning (page 632)], random exploration policy is guaranteed to explore every 
    state/transition a sufficient number of times, but a popular exploration policy that might speed this up is the 
    epsilon-greedy policy, where random actions are made with 1-epsilon probability (thus allowing it to explore unknown
    parts of the problem. MDPToolbox allows for setting a starting epsilon (the epsilon parameter in the constructor), 
    and ending epsilon(epsilon_min) and the decay factor by which to reduce epsilon by each iteration/episode (epsilon_decay)
    This is somewhat similar to simulated annealing.
    Another method is optimism in the face of uncertainty, where Q is initialized to large values.

- Problems
    According to [Geron, Hands on Machine Learning (page 637)], all RL algorithms face 'catastrophic forgetting':
    As the agent explores, it updates its policy but what it learns in a newer environment may harm what was learned earlier.
        * A potential way around this is to reduce the learning rate. In MDPToolBox, this is the alpha parameter. The toolbox
        provides a way to decay this value.
    

- Other Notes
    According to [Mitchell, Machine Learning], Q learning is a off-policy algorithm. It trains a policy, but during runtime
    it doesn't acutally execute that policy, it executes a different policy known as the exploration policy which can
    either be completely random or involve some level of probability in the case of a 1-epsilon exploration strategy.
    
    
"""
def qlearning_experiment(P, R, problem_name, max_iter=10000):
    # for tweaking thresholds based on problem and experiment
    convergences = {
        "Taxi": {
            "Four":{
                "num_iter_req" : 20,
                "thresh": .01,
            },
            "Six": {
                "num_iter_req": 20,
                "thresh": .01,
            },
            "Seven": {
                "num_iter_req": 20,
                "thresh": .01,
            },
            "Eight": {
                "num_iter_req": 20,
                "thresh": .01,
            },
            "Best": {
                "epsilon" : .99,
                "alpha": .99
            }

        },
        "ForestSmall": {
            "Four": {
                "num_iter_req": 40,
                "thresh": .002,
            },
            "Six": {
                "num_iter_req": 40,
                "thresh": .002,
            },
            "Seven": {
                "num_iter_req": 40,
                "thresh": .002,
            },
            "Eight": {
                "num_iter_req": 40,
                "thresh": .002,
            },
            "Best": {
                "epsilon": .98,
                "alpha": .95
            }
        },
        "ForestMedium": {
            "Four": {
                "num_iter_req": 40,
                "thresh": .002,
            },
            "Six": {
                "num_iter_req": 40,
                "thresh": .002,
            },
            "Seven": {
                "num_iter_req": 40,
                "thresh": .002,
            },
            "Eight": {
                "num_iter_req": 40,
                "thresh": .002,
            },
            "Best": {
                "epsilon": .99,
                "alpha": .95
            }
        },
        "ForestLarge": {
            "Four": {
                "num_iter_req": 40,
                "thresh": .002,
            },
            "Six": {
                "num_iter_req": 40,
                "thresh": .002,
            },
            "Seven": {
                "num_iter_req": 40,
                "thresh": .002,
            },
            "Eight": {
                "num_iter_req": 40,
                "thresh": .002,
            },
            "Best": {
                "epsilon" : .5,
                "alpha": .99
            }
        }
    }
    returned_policy = None
    returned_metrics = {}
    returned_reward = None
    if not os.path.exists('./%s/q_learning' % problem_name):
        os.makedirs('./%s/q_learning' % problem_name)

    # """
    # # Best Run - The best run after tuning
    # """
    if not os.path.exists('./%s/q_learning/best_run' % problem_name):
        os.makedirs('./%s/q_learning/best_run' % problem_name)
    ql = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=.99, epsilon=convergences[problem_name]["Best"]["epsilon"], epsilon_decay=.99, n_iter=1000000, alpha=convergences[problem_name]["Best"]["alpha"])

    data = ql.run()
    returned_policy = ql.policy
    states, actions, rewards, error, time,\
    alpha, epsilon, gamma, maxV, meanV, iteration = q_learn_stats_to_tuple(data)
    returned_reward = np.max(rewards)
    error_converge = thresh_convergence_test(np.array(error),.05)

    plot_single(x=iteration,y=time,title="Q Learning - %s - Time Over Iteration" % problem_name,
                fn = "%s/q_learning/best_run/iteration_time" % (problem_name),xlab="Iteration",ylab= "Time (s)",
                convergenceX=error_converge)

    plot_single(x=iteration,y=error,title="Q Learning - %s - Error Over Iteration" % problem_name,
                fn = "%s/q_learning/best_run/iteration_error" % (problem_name),xlab="Iteration",ylab= "Error",
                convergenceX=error_converge)

    plot_single(x=iteration,y=epsilon,title="Q Learning - %s - Epsilon Over Iteration" % problem_name,
                fn = "%s/q_learning/best_run/iteration_epsilon" % (problem_name),xlab="Iteration",ylab= "Epsilon")

    plot_single(x=iteration,y=rewards,title="Q Learning - %s - Reward Over Iteration" % problem_name,
                fn = "%s/q_learning/best_run/iteration_reward" % (problem_name),xlab="Iteration",ylab= "Reward")

    plot_single(x=iteration,y=alpha,title="Q Learning - %s - Alpha Over Iteration" % problem_name,
                fn = "%s/q_learning/best_run/iteration_alpha" % (problem_name),xlab="Iteration",ylab= "Alpha")

    plot_single(x=iteration,y=maxV,title="Q Learning - %s - maxV Over Iteration" % problem_name,
                fn = "%s/q_learning/best_run/iteration_v_max" % (problem_name),xlab="Iteration",ylab= "V_Max")

    plot_single(x=iteration,y=meanV,title="Q Learning - %s - meanV Over Iteration" % problem_name,
                fn = "%s/q_learning/best_run/iteration_v_mean" % (problem_name),xlab="Iteration",ylab= "V_Mean")

    generate_statemap(states,P.shape[1], "%s/q_learning/best_run/statemap" % (problem_name))



    """
    # Fourth Run - What exploration strategies did you choose? (Continued) - Constant Learning Rate Using Multiple Constants
    We can expand on the constant approach to the learning rate by running multiple Q learning attempts over several possible
    QLearn attempts and note the convergence behavior across all of them. It is important to note that reducing the learning
    may help reduce the chance of catastrophic forgetting, which might be beneficial if the 'sweet spot' can be found.
    This helps identify an intial learning rate as well.
    """
    print("EXP 4")
    if not os.path.exists('./%s/q_learning/fourth_run' % problem_name):
        os.makedirs('./%s/q_learning/fourth_run' % problem_name)

    convergence_points = []
    convergence_rewards = []
    err_arr = []
    xs = []
    convergence_runtimes = []
    for i in range(1,100,5):
        ql = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=.9, n_iter=max_iter, alpha_decay=.99, alpha=i/100)
        data = ql.run()
        states, actions, rewards, error, time,\
        alpha, epsilon, gamma, maxV, meanV, iteration = q_learn_stats_to_tuple(data)

        xs.append(i / 100)
        # convergences = {
        #     "Taxi": {
        #         "Four": {
        #             "num_iter_req": 20,
        #             "thresh": .01,
        #         },
        error_converge = thresh_convergence_test(np.array(error), convergences[problem_name]["Four"]["thresh"],convergences[problem_name]["Four"]["num_iter_req"])
        convergence_points.append(error_converge)
        err_arr.append(error)
        convergence_runtimes.append(time[error_converge-1])
        convergence_rewards.append(rewards[error_converge-1])
        plot_single(x=iteration,y=error,title="Q Learning - %s - Error Over Iteration" % problem_name,
                    fn = "%s/q_learning/fourth_run/iteration_error_%d" % (problem_name, i),xlab="Iteration",ylab= "Error",
                    convergenceX=error_converge)
        plot_single(x=iteration,y=rewards,title="Q Learning - %s - Reward Over Iteration" % problem_name,
                    fn = "%s/q_learning/fourth_run/iteration_reward_%d" % (problem_name, i),xlab="Iteration",ylab= "Error",
                    convergenceX=error_converge)
    returned_metrics["fourth_run"] = {
        "xs": xs,
        "convergence_runtimes": convergence_runtimes,
        "convergence_rewards": convergence_rewards,
        "convergence_points": convergence_points,
        "error": err_arr
    }
    plot_single(x=xs,y=convergence_points,title="Q Learning - %s - Iterations To Converge As Alpha Changes" % problem_name,
                fn = "%s/q_learning/fourth_run/alpha_v_convergence" % (problem_name),xlab="Alpha",ylab= "Convergence Iteration")
    plot_single(x=xs,y=convergence_rewards,title="Q Learning - %s - Reward As Alpha Changes" % problem_name,
                fn = "%s/q_learning/fourth_run/alpha_v_reward" % (problem_name),xlab="Alpha",ylab= "Reward")
    plot_single(x=xs,y=convergence_runtimes,title="Q Learning - %s - Runtime As Alpha Changes" % problem_name,
                fn = "%s/q_learning/fourth_run/alpha_v_runtime" % (problem_name),xlab="Alpha",ylab= "Runtime")

    """
    # Sixth Run - Need to identify an ideal initial epsilon
    """
    print("EXP 6")
    if not os.path.exists('./%s/q_learning/sixth_run' % problem_name):
        os.makedirs('./%s/q_learning/sixth_run' % problem_name)
    convergence_points = []
    convergence_rewards = []
    err_arr = []
    xs = []
    convergence_runtimes = []
    for i in range(1, 100, 5):
        ql = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=.9, epsilon=i/100, epsilon_decay=1, n_iter=max_iter)
        ql.setVerbose()
        data = ql.run()
        states, actions, rewards, error, time, \
        alpha, epsilon, gamma, maxV, meanV, iteration = q_learn_stats_to_tuple(data)

        xs.append(i / 100)
        error_converge = thresh_convergence_test(np.array(error), convergences[problem_name]["Six"]["thresh"],convergences[problem_name]["Six"]["num_iter_req"])
        convergence_points.append(error_converge)
        err_arr.append(error)
        convergence_runtimes.append(time[error_converge-1])
        convergence_rewards.append(rewards[error_converge-1])
        plot_single(x=iteration,y=error,title="Q Learning - %s - Error Over Iteration" % problem_name,
                    fn = "%s/q_learning/sixth_run/iteration_error_%d" % (problem_name, i),xlab="Iteration",ylab= "Error",
                    convergenceX=error_converge)

    returned_metrics["sixth_run"] = {
        "xs": xs,
        "convergence_runtimes": convergence_runtimes,
        "convergence_rewards": convergence_rewards,
        "convergence_points": convergence_points,
        "error": err_arr
    }

    plot_single(x=xs,y=convergence_points,title="Q Learning - %s - Iterations To Converge As Initial Epsilon Changes" % problem_name,
                fn = "%s/q_learning/sixth_run/epsilon_init_v_convergence" % (problem_name),xlab="Initial Epsilon",ylab= "Convergence Iteration")
    plot_single(x=xs,y=convergence_rewards,title="Q Learning - %s - Reward As Epsilon Changes" % problem_name,
                fn = "%s/q_learning/sixth_run/epsilon_v_reward" % (problem_name),xlab="Epsilon",ylab= "Reward")
    plot_single(x=xs,y=convergence_runtimes,title="Q Learning - %s - Runtime As Epsilon Changes" % problem_name,
                fn = "%s/q_learning/sixth_run/epsilon_v_runtime" % (problem_name),xlab="Epsilon",ylab= "Runtime")
    """
    # Seventh Run - Need to identify an ideal epsilon decay.
    """
    print("EXP 7")
    if not os.path.exists('./%s/q_learning/seventh_run' % problem_name):
        os.makedirs('./%s/q_learning/seventh_run' % problem_name)
    convergence_points = []
    convergence_rewards = []
    err_arr = []
    xs = []
    convergence_runtimes = []
    for i in range(1, 100, 5):
        ql = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=.9, epsilon=1, epsilon_decay=i/100, n_iter=max_iter, alpha=.95)
        ql.setVerbose()
        data = ql.run()
        states, actions, rewards, error, time, \
        alpha, epsilon, gamma, maxV, meanV, iteration = q_learn_stats_to_tuple(data)

        xs.append(i / 100)
        error_converge = thresh_convergence_test(np.array(error), convergences[problem_name]["Seven"]["thresh"],convergences[problem_name]["Seven"]["num_iter_req"])
        convergence_points.append(error_converge)
        err_arr.append(error)
        convergence_runtimes.append(time[error_converge-1])
        convergence_rewards.append(rewards[error_converge-1])
        plot_single(x=iteration,y=error,title="Q Learning - %s - Error Over Iteration" % problem_name,
                    fn = "%s/q_learning/seventh_run/iteration_error_%d" % (problem_name, i),xlab="Iteration",ylab= "Error",
                    convergenceX=error_converge)

    returned_metrics["seventh_run"] = {
        "xs": xs,
        "convergence_runtimes": convergence_runtimes,
        "convergence_rewards": convergence_rewards,
        "convergence_points": convergence_points,
        "error": err_arr
    }
    plot_single(x=xs,y=convergence_points,title="Q Learning - %s - Iterations To Converge As Epsilon Decay Changes" % problem_name,
                fn = "%s/q_learning/seventh_run/epsilon_decay_v_convergence" % (problem_name),xlab="Epsilon Decay",ylab= "Convergence Iteration")
    plot_single(x=xs, y=convergence_rewards, title="Q Learning - %s - Reward As Epsilon Decay Changes" % problem_name,
                fn="%s/q_learning/seventh_run/epsilondecay_v_reward" % (problem_name), xlab="Epsilon Decay",
                ylab="Reward")
    plot_single(x=xs, y=convergence_runtimes, title="Q Learning - %s - Runtime As Epsilon Decay Changes" % problem_name,
                fn="%s/q_learning/seventh_run/epsilondecay_v_runtime" % (problem_name), xlab="Epsilon Decay",
                ylab="Runtime")
    """
    # Eighth Run - Need to identify an ideal gamma/discount factor
    """
    print("EXP 8")
    if not os.path.exists('./%s/q_learning/eighth_run' % problem_name):
        os.makedirs('./%s/q_learning/eighth_run' % problem_name)
    convergence_points = []
    convergence_rewards = []
    err_arr = []
    xs = []
    convergence_runtimes = []
    for i in range(1, 100, 5):
        ql = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=i/100, n_iter=max_iter)
        ql.setVerbose()
        data = ql.run()
        states, actions, rewards, error, time, \
        alpha, epsilon, gamma, maxV, meanV, iteration = q_learn_stats_to_tuple(data)

        xs.append(i / 100)
        error_converge = thresh_convergence_test(np.array(error), convergences[problem_name]["Eight"]["thresh"],convergences[problem_name]["Eight"]["num_iter_req"])
        convergence_points.append(error_converge)
        err_arr.append(error)
        convergence_runtimes.append(time[error_converge-1])
        convergence_rewards.append(rewards[error_converge-1])
        plot_single(x=iteration,y=error,title="Q Learning - %s - Error Over Iteration" % problem_name,
                    fn = "%s/q_learning/eighth_run/iteration_error_%d" % (problem_name, i),xlab="Iteration",ylab= "Error",
                    convergenceX=error_converge)

    returned_metrics["eighth_run"] = {
        "xs": xs,
        "convergence_runtimes": convergence_runtimes,
        "convergence_rewards": convergence_rewards,
        "convergence_points": convergence_points,
        "error": err_arr
    }

    plot_single(x=xs,y=convergence_points,title="Q Learning - %s - Iterations To Converge As Gamma Changes" % problem_name,
                fn = "%s/q_learning/eighth_run/gamma_v_convergence" % (problem_name),xlab="Gamma",ylab= "Convergence Iteration")
    plot_single(x=xs, y=convergence_rewards, title="Q Learning - %s - Reward As Gamma Changes" % problem_name,
                fn="%s/q_learning/eighth_run/gamma_v_reward" % (problem_name), xlab="Gamma", ylab="Reward")
    plot_single(x=xs, y=convergence_runtimes, title="Q Learning - %s - Runtime As Gamma Changes" % problem_name,
                fn="%s/q_learning/eighth_run/gamma_v_runtime" % (problem_name), xlab="Gamma", ylab="Runtime")


    print("Q Learning Run Complete")
    return returned_policy, returned_reward, returned_metrics


"""
Given a series, a lower bound value, and a minimum number of iterations, return the index where the convergence occurs
which is defined by this function as the point which the series drops below the lower bound given and stays within that
lower bound for the given minimal number of iterations. Returns -1 for no convergence. If None is give for lower bound,
the convergence will be based on the largest run of consecutive numbers below 10 percent of the max
"""
def thresh_convergence_test(series, lower_bound, min_num_iterations = 20):
    data = np.asanyarray(np.where(series < lower_bound)).flatten()
    x = np.split(data, np.where(np.diff(data) != 1)[0]+1)
    for group in x:
        if(len(group) >= min_num_iterations):
            return group[0]
    return -1


"""
Convert QLearn stats dictionary into a tuple of arrays.
"""
def q_learn_stats_to_tuple(data):
    states, actions, rewards, error, time,\
    alpha, epsilon, gamma, maxV, meanV, iteration = [], [], [], [], [], [], [], [], [], [], []
    for stat in data:
        states.append(stat["State"])
        actions.append(stat["Action"])
        rewards.append(stat["Reward"])
        error.append(stat["Error"])
        time.append(stat["Time"])
        alpha.append(stat["Alpha"])
        epsilon.append(stat["Epsilon"])
        gamma.append(stat["Gamma"])
        maxV.append(stat["Max V"])
        meanV.append(stat["Mean V"])
        iteration.append(stat["Iteration"])
    return states, actions, rewards, error, time, alpha, epsilon, gamma, maxV, meanV, iteration

"""
Convert Policy Iteration stats dictionary into a tuple of arrays.
"""
def policy_value_iteration_stats_to_tuple(data):
    states, actions, rewards, error, time, maxV, meanV, iteration = [], [], [], [], [], [], [], []
    for stat in data:
        states.append(stat["State"])
        actions.append(stat["Action"])
        rewards.append(stat["Reward"])
        error.append(stat["Error"])
        time.append(stat["Time"])
        maxV.append(stat["Max V"])
        meanV.append(stat["Mean V"])
        iteration.append(stat["Iteration"])
    return states, actions, rewards, error, time, maxV, meanV, iteration

"""
Visualization of states visited and frequency.
"""
def generate_statemap(states, num_states, output):
    sqt = int(np.ceil(np.sqrt(num_states)))
    arr = np.zeros((sqt,sqt)) # create square matrix
    for state in states:
        row = int(state/sqt)
        col = int(state%sqt)
        arr[row,col] = arr[row,col]+1

    sns.heatmap(arr, square=True)
    plt.xlim(0, arr.shape[0])
    plt.ylim(0, arr.shape[1])
    plt.savefig(output)
    plt.close()


"""
Runs MDP Problem - Forest
"""
def run_mdp_forest(run_value_iteration = True, run_policy_iteration = False, run_q_learning = False):


    if not os.path.exists('./%s' % "Forest"):
        os.makedirs('./%s' % "Forest")

    # define problem
    P_Small, R_Small = hiive.mdptoolbox.example.forest(S=100, p=.01)
    P_Medium, R_Medium = hiive.mdptoolbox.example.forest(S=900, p=.01)
    P_Large, R_Large = hiive.mdptoolbox.example.forest(S=3600, p=.01)

    if run_value_iteration:
        ###########################################################
        #Value iteration
        #########################################################
        '''
        Primary experiments - Forest Small
        '''
        #def run_small_vi(P_Small,R_Small):
        problem_name = "ForestSmall"
        if not os.path.exists('./%s' % problem_name):
            os.makedirs('./%s' % problem_name)
        P = P_Small
        R= R_Small
        policy_vi_Small, reward_vi_Small, metrics_vi_Small = value_iteration_experiment(P,R,problem_name)
        generate_policy_map(policy_vi_Small, problem_name, "./%s/vi_%s_policy_Small" % (problem_name,problem_name))
            #return policy_vi_Small, reward_vi_Small, metrics_vi_Small



        '''
        Primary experiments - forest Medium
        '''

        #def run_medium_vi(P_Medium, R_Medium):
        problem_name = "ForestMedium"
        if not os.path.exists('./%s' % problem_name):
            os.makedirs('./%s' % problem_name)
        P = P_Medium
        R= R_Medium
        policy_vi_Medium, reward_vi_Medium, metrics_vi_Medium = value_iteration_experiment(P,R,problem_name)
        generate_policy_map(policy_vi_Medium, problem_name, "./%s/vi_%s_policy_Medium" % (problem_name,problem_name), annotate=False)
            #return policy_vi_Medium, reward_vi_Medium, metrics_vi_Medium

        '''
        Primary experiments - forest Large
        '''

        #def run_large_vi(P_Large, R_Large):
        problem_name = "ForestLarge"
        if not os.path.exists('./%s' % problem_name):
            os.makedirs('./%s' % problem_name)
        P = P_Large
        R= R_Large
        policy_vi_Large, reward_vi_Large, metrics_vi_Large = value_iteration_experiment(P,R,problem_name)
        generate_policy_map(policy_vi_Large, problem_name, "./%s/vi_%s_policy_Large" % (problem_name,problem_name), annotate=False)
        #return policy_vi_Large, reward_vi_Large, metrics_vi_Large

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     future_small = executor.submit(run_small_vi, P_Small, R_Small)
        #     policy_vi_Small, reward_vi_Small, metrics_vi_Small = future_small.result()
        #     future_medium = executor.submit(run_medium_vi, P_Medium, R_Medium)
        #     policy_vi_Medium, reward_vi_Medium, metrics_vi_Medium = future_medium.result()
        #     future_large = executor.submit(run_large_vi, P_Large, R_Large)
        #     policy_vi_Large, reward_vi_Large, metrics_vi_Large = future_large.result()



        '''
        Primary experiments - combined metrics
        '''
        # "convergence_runtimes": convergence_runtimes,
        # "convergence_rewards": convergence_rewards,
        # "convergence_points": convergence_points

        # Value Iteration - vary gamma
        # Draw lines
        df = pd.DataFrame({"xs": metrics_vi_Small["second_run"]["xs"],"Forest-Small": metrics_vi_Small["second_run"]["convergence_points"], "Forest-Medium": metrics_vi_Medium["second_run"]["convergence_points"], "Forest-Large": metrics_vi_Large["second_run"]["convergence_points"]})
        # plt.plot(df, linewidth=2)
        # df.plot(kind='line')
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # Create plot
        plt.title("Convergence Iteration Over Changing Gamma - Forest")
        plt.xlabel("Gamma"), plt.ylabel("Convergence Iteration")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/convergence_v_gamma_all_sizes_vi")
        plt.close()

        df = pd.DataFrame({"xs": metrics_vi_Small["second_run"]["xs"],"Forest-Small": metrics_vi_Small["second_run"]["convergence_rewards"], "Forest-Medium": metrics_vi_Medium["second_run"]["convergence_rewards"], "Forest-Large": metrics_vi_Large["second_run"]["convergence_rewards"]})

        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        #df.plot(kind='line')
        # Create plot
        plt.title("Reward Over Changing Gamma - Forest")
        plt.xlabel("Gamma"), plt.ylabel("Reward")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/reward_v_gamma_all_sizes_vi")
        plt.close()

        df = pd.DataFrame({"xs": metrics_vi_Small["second_run"]["xs"], "Forest-Small": metrics_vi_Small["second_run"]["convergence_runtimes"], "Forest-Medium": metrics_vi_Medium["second_run"]["convergence_runtimes"], "Forest-Large": metrics_vi_Large["second_run"]["convergence_runtimes"]})
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        #df.plot(kind='line')
        # Create plot
        plt.title("Runtimes Over Changing Gamma - Forest")
        plt.xlabel("Gamma"), plt.ylabel("Runtime")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/runtime_v_gamma_all_sizes_vi")
        plt.close()


        # Value Iteration - vary epsilon
        # Draw lines
        df = pd.DataFrame({"xs": metrics_vi_Small["third_run"]["xs"], "Forest-Small": metrics_vi_Small["third_run"]["convergence_points"], "Forest-Medium": metrics_vi_Medium["third_run"]["convergence_points"], "Forest-Large": metrics_vi_Large["third_run"]["convergence_points"]})
        # plt.plot(df, linewidth=2)
        # df.plot()
        # df.plot(kind='line')
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # Create plot
        plt.title("Convergence Iteration Over Changing Epsilon - Forest")
        plt.xlabel("Epsilon"), plt.ylabel("Convergence Iteration")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/convergence_v_epsilon_all_sizes_vi")
        plt.close()

        df = pd.DataFrame({"xs": metrics_vi_Small["third_run"]["xs"], "Forest-Small": metrics_vi_Small["third_run"]["convergence_rewards"], "Forest-Medium": metrics_vi_Medium["third_run"]["convergence_rewards"], "Forest-Large": metrics_vi_Large["third_run"]["convergence_rewards"]})
        # plt.plot(df, linewidth=2)
        # df.plot()
        # df.plot(kind='line')
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # Create plot
        plt.title("Reward Over Changing Epsilon - Forest")
        plt.xlabel("Epsilon"), plt.ylabel("Reward")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/reward_v_epsilon_all_sizes_vi")
        plt.close()

        df = pd.DataFrame({"xs": metrics_vi_Small["third_run"]["xs"], "Forest-Small": metrics_vi_Small["third_run"]["convergence_runtimes"], "Forest-Medium": metrics_vi_Medium["third_run"]["convergence_runtimes"], "Forest-Large": metrics_vi_Large["third_run"]["convergence_runtimes"]})
        # plt.plot(df, linewidth=2)
        # df.plot()
        # df.plot(kind='line')
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # Create plot
        plt.title("Runtimes Over Changing Epsilon - Forest")
        plt.xlabel("Gamma"), plt.ylabel("Runtime")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/runtime_v_epsilon_all_sizes_vi")
        plt.close()

    if run_policy_iteration:
        ###########################################################
        #Policy iteration
        #########################################################
        '''
        Primary experiments - Forest Small
        '''
        problem_name = "ForestSmall"
        if not os.path.exists('./%s' % problem_name):
            os.makedirs('./%s' % problem_name)
        P = P_Small
        R = R_Small
        policy_pi_Small, reward_pi_Small, metrics_pi_Small = policy_iteration_experiment(P,R,problem_name)
        generate_policy_map(policy_pi_Small, problem_name, "./%s/pi_%s_policy_Small" % (problem_name,problem_name))

        '''
        Primary experiments - forest Medium
        '''
        problem_name = "ForestMedium"
        if not os.path.exists('./%s' % problem_name):
            os.makedirs('./%s' % problem_name)
        P = P_Medium
        R = R_Medium
        policy_pi_Medium, reward_pi_Medium, metrics_pi_Medium = policy_iteration_experiment(P, R, problem_name)
        generate_policy_map(policy_pi_Medium, problem_name, "./%s/pi_%s_policy_Medium" % (problem_name, problem_name),
                            annotate=False)

        '''
        Primary experiments - forest Large
        '''
        problem_name = "ForestLarge"
        if not os.path.exists('./%s' % problem_name):
            os.makedirs('./%s' % problem_name)
        P = P_Large
        R = R_Large
        policy_pi_Large, reward_pi_Large, metrics_pi_Large = policy_iteration_experiment(P, R, problem_name)
        generate_policy_map(policy_pi_Large, problem_name, "./%s/pi_%s_policy_Large" % (problem_name, problem_name),
                            annotate=False)


        '''
        Primary experiments - combined metrics
        '''
        # "convergence_runtimes": convergence_runtimes,
        # "convergence_rewards": convergence_rewards,
        # "convergence_points": convergence_points

        # Value Iteration - vary gamma
        # Draw lines
        df = pd.DataFrame({"xs": metrics_pi_Small["second_run"]["xs"],
                           "Forest-Small": metrics_pi_Small["second_run"]["convergence_points"],
                           "Forest-Medium": metrics_pi_Medium["second_run"]["convergence_points"],
                           "Forest-Large": metrics_pi_Large["second_run"]["convergence_points"]})
        # plt.plot(df, linewidth=2)
        # df.plot(kind='line')
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # Create plot
        plt.title("Convergence Iteration Over Changing Gamma - Forest")
        plt.xlabel("Gamma"), plt.ylabel("Convergence Iteration")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/convergence_v_gamma_all_sizes_pi")
        plt.close()

        df = pd.DataFrame({"xs": metrics_pi_Small["second_run"]["xs"],
                           "Forest-Small": metrics_pi_Small["second_run"]["convergence_rewards"],
                           "Forest-Medium": metrics_pi_Medium["second_run"]["convergence_rewards"],
                           "Forest-Large": metrics_pi_Large["second_run"]["convergence_rewards"]})

        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # df.plot(kind='line')
        # Create plot
        plt.title("Reward Over Changing Gamma - Forest")
        plt.xlabel("Gamma"), plt.ylabel("Reward")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/reward_v_gamma_all_sizes_pi")
        plt.close()

        df = pd.DataFrame({"xs": metrics_pi_Small["second_run"]["xs"],
                           "Forest-Small": metrics_pi_Small["second_run"]["convergence_runtimes"],
                           "Forest-Medium": metrics_pi_Medium["second_run"]["convergence_runtimes"],
                           "Forest-Large": metrics_pi_Large["second_run"]["convergence_runtimes"]})
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # df.plot(kind='line')
        # Create plot
        plt.title("Runtimes Over Changing Gamma - Forest")
        plt.xlabel("Gamma"), plt.ylabel("Runtime")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/runtime_v_gamma_all_sizes_pi")
        plt.close()

        # Policy Iteration - vary epsilon
        # Draw lines
        df = pd.DataFrame({"xs": metrics_pi_Small["third_run"]["xs"],
                           "Forest-Small": metrics_pi_Small["third_run"]["convergence_points"],
                           "Forest-Medium": metrics_pi_Medium["third_run"]["convergence_points"],
                           "Forest-Large": metrics_pi_Large["third_run"]["convergence_points"]})
        # plt.plot(df, linewidth=2)
        # df.plot()
        # df.plot(kind='line')
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # Create plot
        plt.title("Convergence Iteration Over Changing Epsilon - Forest")
        plt.xlabel("Epsilon"), plt.ylabel("Convergence Iteration")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/convergence_v_epsilon_all_sizes_pi")
        plt.close()

        df = pd.DataFrame({"xs": metrics_pi_Small["third_run"]["xs"],
                           "Forest-Small": metrics_pi_Small["third_run"]["convergence_rewards"],
                           "Forest-Medium": metrics_pi_Medium["third_run"]["convergence_rewards"],
                           "Forest-Large": metrics_pi_Large["third_run"]["convergence_rewards"]})
        # plt.plot(df, linewidth=2)
        # df.plot()
        # df.plot(kind='line')
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # Create plot
        plt.title("Reward Over Changing Epsilon - Forest")
        plt.xlabel("Epsilon"), plt.ylabel("Reward")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/reward_v_epsilon_all_sizes_pi")
        plt.close()

        df = pd.DataFrame({"xs": metrics_pi_Small["third_run"]["xs"],
                           "Forest-Small": metrics_pi_Small["third_run"]["convergence_runtimes"],
                           "Forest-Medium": metrics_pi_Medium["third_run"]["convergence_runtimes"],
                           "Forest-Large": metrics_pi_Large["third_run"]["convergence_runtimes"]})
        # plt.plot(df, linewidth=2)
        # df.plot()
        # df.plot(kind='line')
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # Create plot
        plt.title("Runtimes Over Changing Epsilon - Forest")
        plt.xlabel("Gamma"), plt.ylabel("Runtime")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/runtime_v_epsilon_all_sizes_pi")
        plt.close()

    if run_q_learning:
        '''
        Primary experiments - q learning
        '''
        problem_name = "ForestSmall"
        if not os.path.exists('./%s' % problem_name):
            os.makedirs('./%s' % problem_name)
        P = P_Small
        R = R_Small
        policy_q_Small, reward_q_Small, metrics_q_Small = qlearning_experiment(P,R,problem_name)
        if policy_q_Small is not None:
            generate_policy_map(policy_q_Small, problem_name, "./%s/q_%s_policy_Small" % (problem_name,problem_name))

        '''
        Primary experiments - forest Medium
        '''
        problem_name = "ForestMedium"
        if not os.path.exists('./%s' % problem_name):
            os.makedirs('./%s' % problem_name)
        P = P_Medium
        R = R_Medium
        policy_q_Medium, reward_q_Medium, metrics_q_Medium = qlearning_experiment(P, R, problem_name)
        if policy_q_Medium is not None:
            generate_policy_map(policy_q_Medium, problem_name, "./%s/q_%s_policy_Medium" % (problem_name, problem_name),
                                annotate=False)

        '''
        Primary experiments - forest Large
        '''
        problem_name = "ForestLarge"
        if not os.path.exists('./%s' % problem_name):
            os.makedirs('./%s' % problem_name)
        P = P_Large
        R = R_Large
        policy_q_Large, reward_q_Large, metrics_q_Large = qlearning_experiment(P, R, problem_name)
        if policy_q_Large is not None:
            generate_policy_map(policy_q_Large, problem_name, "./%s/q_%s_policy_Large" % (problem_name, problem_name),
                                annotate=False)


        df = pd.DataFrame({"xs": metrics_q_Small["fourth_run"]["xs"],
                           "Forest-Small": metrics_q_Small["fourth_run"]["convergence_points"],
                           "Forest-Medium": metrics_q_Medium["fourth_run"]["convergence_points"],
                           "Forest-Large": metrics_q_Large["fourth_run"]["convergence_points"]})
        # plt.plot(df, linewidth=2)
        # df.plot(kind='line')
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # Create plot
        plt.title("Convergence Iteration Over Changing Learning Rate - Forest")
        plt.xlabel("Alpha"), plt.ylabel("Convergence Iteration")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/convergence_v_alpha_all_sizes_q")
        plt.close()

        df = pd.DataFrame({"xs": metrics_q_Small["fourth_run"]["xs"],
                           "Forest-Small": metrics_q_Small["fourth_run"]["convergence_runtimes"],
                           "Forest-Medium": metrics_q_Medium["fourth_run"]["convergence_runtimes"],
                           "Forest-Large": metrics_q_Large["fourth_run"]["convergence_runtimes"]})
        # plt.plot(df, linewidth=2)
        # df.plot(kind='line')
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # Create plot
        plt.title("Runtime Over Changing Learning Rate - Forest")
        plt.xlabel("Alpha"), plt.ylabel("Runtime")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/runtime_v_alpha_all_sizes_q")
        plt.close()

        df = pd.DataFrame({"xs": metrics_q_Small["sixth_run"]["xs"],
                           "Forest-Small": metrics_q_Small["sixth_run"]["convergence_points"],
                           "Forest-Medium": metrics_q_Medium["sixth_run"]["convergence_points"],
                           "Forest-Large": metrics_q_Large["sixth_run"]["convergence_points"]})
        # plt.plot(df, linewidth=2)
        # df.plot(kind='line')
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # Create plot
        plt.title("Convergence Iteration Over Changing Initial Epsilon - Forest")
        plt.xlabel("Initial Epsilon"), plt.ylabel("Convergence Iteration")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/convergence_v_init_epsilon_all_sizes_q")
        plt.close()

        df = pd.DataFrame({"xs": metrics_q_Small["sixth_run"]["xs"],
                           "Forest-Small": metrics_q_Small["sixth_run"]["convergence_runtimes"],
                           "Forest-Medium": metrics_q_Medium["sixth_run"]["convergence_runtimes"],
                           "Forest-Large": metrics_q_Large["sixth_run"]["convergence_runtimes"]})
        # plt.plot(df, linewidth=2)
        # df.plot(kind='line')
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # Create plot
        plt.title("Runtime Over Changing Initial Epsilon - Forest")
        plt.xlabel("Initial Epsilon"), plt.ylabel("Runtime")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/runtime_v_init_epsilon_all_sizes_q")
        plt.close()

        df = pd.DataFrame({"xs": metrics_q_Small["seventh_run"]["xs"],
                           "Forest-Small": metrics_q_Small["seventh_run"]["convergence_points"],
                           "Forest-Medium": metrics_q_Medium["seventh_run"]["convergence_points"],
                           "Forest-Large": metrics_q_Large["seventh_run"]["convergence_points"]})
        # plt.plot(df, linewidth=2)
        # df.plot(kind='line')
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # Create plot
        plt.title("Convergence Iteration Over Changing Epsilon Decay - Forest")
        plt.xlabel("Epsilon Decay"), plt.ylabel("Convergence Iteration")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/convergence_v_decay_epsilon_all_sizes_q")
        plt.close()

        df = pd.DataFrame({"xs": metrics_q_Small["eighth_run"]["xs"],
                           "Forest-Small": metrics_q_Small["eighth_run"]["convergence_points"],
                           "Forest-Medium": metrics_q_Medium["eighth_run"]["convergence_points"],
                           "Forest-Large": metrics_q_Large["eighth_run"]["convergence_points"]})
        # plt.plot(df, linewidth=2)
        # df.plot(kind='line')
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # Create plot
        plt.title("Convergence Iteration Over Changing Gamma - Forest")
        plt.xlabel("Gamma"), plt.ylabel("Convergence Iteration")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/convergence_v_gamma_all_sizes_q")
        plt.close()

        df = pd.DataFrame({"xs": metrics_q_Small["eighth_run"]["xs"],
                           "Forest-Small": metrics_q_Small["eighth_run"]["convergence_runtimes"],
                           "Forest-Medium": metrics_q_Medium["eighth_run"]["convergence_runtimes"],
                           "Forest-Large": metrics_q_Large["eighth_run"]["convergence_runtimes"]})
        # plt.plot(df, linewidth=2)
        # df.plot(kind='line')
        plt.plot('xs', "Forest-Small", linewidth=2, data=df)
        plt.plot('xs', "Forest-Medium", linewidth=2, data=df)
        plt.plot('xs', "Forest-Large", linewidth=2, data=df)
        # Create plot
        plt.title("Runtime Over Changing Gamma - Forest")
        plt.xlabel("Gamma"), plt.ylabel("Runtime")
        plt.tight_layout()
        plt.legend()
        plt.savefig("./Forest/runtime_v_gamma_all_sizes_q")
        plt.close()

    print("Completed Forest Experiment")

"""
Converts a gym environment to P,R matrices used by mdptoolbox
"""
def gym_env_to_p_r(env):
    # found on kaggle, this code is credited to a former student: Blake Wang
    env.reset()
    nA, nS = env.nA, env.nS
    P_fl = np.zeros([nA, nS, nS])
    R_fl = np.zeros([nS, nA])
    for s in range(nS):
        for a in range(nA):
            transitions = env.P[s][a]
            for p_trans, next_s, reward, _ in transitions:
                P_fl[a, s, next_s] += p_trans
                R_fl[s, a] = reward
            P_fl[a, s, :] /= np.sum(P_fl[a, s, :])
    return P_fl, R_fl

"""
Runs a gym environent by the given env_name. problem_name is used for labeling charts and creating the folder and can
be anything.
"""
def run_mdp_gym(run_value_iteration = True, run_policy_iteration = False, run_q_learning = False):
    problem_name, env_name = "Taxi", 'Taxi-v3'
    env = gym.make(env_name)
    P, R = gym_env_to_p_r(env)
    if run_value_iteration:
        '''
        Primary experiments - value iteration
        '''
        policy_vi, reward_vi, metrics_vi = value_iteration_experiment(P,R,problem_name)
        generate_policy_map(policy_vi, problem_name, "./%s/vi_%s_policy" % (problem_name,problem_name))
        #
        # '''
        # Primary experiments - combined metrics
        # '''
        #
        # # Value Iteration - vary gamma
        # plot_single(metrics_vi["second_run"]["xs"], metrics_vi["second_run"]["convergence_points"],
        #             "./%s/convergence_v_gamma_all_sizes_vi" % problem_name,
        #             "Convergence Iteration Over Changing Gamma - %s" % problem_name, xlab="Gamma",
        #             ylab="Convergence Iteration")
        # plot_single(metrics_vi["second_run"]["xs"], metrics_vi["second_run"]["convergence_runtimes"],
        #             "./%s/runtime_v_gamma_all_sizes_vi" % problem_name,
        #             "Runtimes Over Changing Gamma - %s" % problem_name, xlab="Gamma",
        #             ylab="Runtime")
        # plot_single(metrics_vi["second_run"]["xs"], metrics_vi["second_run"]["convergence_rewards"],
        #             "./%s/reward_v_gamma_all_sizes_vi" % problem_name,
        #             "Reward Over Changing Gamma - %s" % problem_name, xlab="Gamma",
        #             ylab="Reward")
        #
        # # Value Iteration - vary epsilon
        # plot_single(metrics_vi["third_run"]["xs"], metrics_vi["third_run"]["convergence_points"],
        #             "./%s/convergence_v_epsilon_all_sizes_vi" % problem_name,
        #             "Convergence Iteration Over Changing Epsilon - %s" % problem_name, xlab="Epsilon",
        #             ylab="Convergence Iteration")
        # plot_single(metrics_vi["third_run"]["xs"], metrics_vi["third_run"]["convergence_runtimes"],
        #             "./%s/runtime_v_epsilon_all_sizes_vi" % problem_name,
        #             "Runtimes Over Changing Epsilon - %s" % problem_name, xlab="Epsilon",
        #             ylab="Runtime")
        # plot_single(metrics_vi["third_run"]["xs"], metrics_vi["third_run"]["convergence_rewards"],
        #             "./%s/reward_v_epsilon_all_sizes_vi" % problem_name,
        #             "Reward Over Changing Epsilon - %s" % problem_name, xlab="Epsilon",
        #             ylab="Reward")

    if run_policy_iteration:
        '''
        Primary experiments - policy iteration
        '''
        policy_pi, reward_pi, metrics_pi = policy_iteration_experiment(P,R,problem_name)
        generate_policy_map(policy_pi, problem_name, "./%s/pi_%s_policy" % (problem_name,problem_name))
        # '''
        # Primary experiments - combined metrics
        # '''
        #
        # # Policy Iteration - vary gamma
        # plot_single(metrics_pi["second_run"]["xs"], metrics_pi["second_run"]["convergence_points"],
        #             "./%s/convergence_v_gamma_all_sizes_pi" % problem_name,
        #             "Convergence Iteration Over Changing Gamma - %s" % problem_name, xlab="Gamma",
        #             ylab="Convergence Iteration")
        # plot_single(metrics_pi["second_run"]["xs"], metrics_pi["second_run"]["convergence_runtimes"],
        #             "./%s/runtime_v_gamma_all_sizes_pi" % problem_name,
        #             "Runtimes Over Changing Gamma - %s" % problem_name, xlab="Gamma",
        #             ylab="Runtime")
        # plot_single(metrics_pi["second_run"]["xs"], metrics_pi["second_run"]["convergence_rewards"],
        #             "./%s/reward_v_gamma_all_sizes_pi" % problem_name,
        #             "Reward Over Changing Gamma - %s" % problem_name, xlab="Gamma",
        #             ylab="Reward")

        # # Policy Iteration - vary epsilon
        # plot_single(metrics_pi["third_run"]["xs"], metrics_pi["third_run"]["convergence_points"],
        #             "./%s/convergence_v_epsilon_all_sizes_pi" % problem_name,
        #             "Convergence Iteration Over Changing Epsilon - %s" % problem_name, xlab="Epsilon",
        #             ylab="Convergence Iteration")
        # plot_single(metrics_pi["third_run"]["xs"], metrics_pi["third_run"]["convergence_runtimes"],
        #             "./%s/runtime_v_epsilon_all_sizes_pi" % problem_name,
        #             "Runtimes Over Changing Epsilon - %s" % problem_name, xlab="Epsilon",
        #             ylab="Runtime")
        # plot_single(metrics_pi["third_run"]["xs"], metrics_pi["third_run"]["convergence_rewards"],
        #             "./%s/reward_v_epsilon_all_sizes_pi" % problem_name,
        #             "Reward Over Changing Epsilon - %s" % problem_name, xlab="Epsilon",
        #             ylab="Reward")

    if run_q_learning:
        '''
        Primary experiments - q learning
        '''
        policy_q, reward_q, metrics_q = qlearning_experiment(P,R,problem_name)
        if policy_q is not None:
            generate_policy_map(policy_q, problem_name, "./%s/q_%s_policy" % (problem_name,problem_name))


    print("Completed Taxi Experiment")


def generate_policy_map(policy,name,output, annotate=True):
    num_states = len(policy)
    sqt = int(np.ceil(np.sqrt(num_states)))
    arr = np.zeros((sqt,sqt)) # create square matrix
    arr -= 1
    annot_arr = np.zeros((sqt,sqt),dtype=str) # create square matrix
    syms = get_symbol_map(name)
    for i, pol in enumerate(policy):
        row = int(i/sqt)
        col = int(i%sqt)
        arr[row,col] = pol
        if(annotate):
            annot_arr[row,col] = syms[pol]
    if(annotate):
        sns.heatmap(arr, square=True, annot=annot_arr, fmt="s", linewidths = 1, cmap="inferno", cbar = False)
    else:
        sns.heatmap(arr, square=True, linewidths=1, cmap="inferno", cbar=False)

    plt.xlim(0, arr.shape[0])
    plt.ylim(0, arr.shape[1])
    plt.savefig(output)
    plt.close()


def get_symbol_map(name):
    if name == "Taxi":
        """
            Actions:
            There are 6 discrete deterministic actions:
            - 0: move south
            - 1: move north
            - 2: move east
            - 3: move west
            - 4: pickup passenger
            - 5: drop off passenger
        """
        return ["↓","↑","→","←","P","D"]
    if name == "Forest" or name == "ForestSmall" or name == "ForestMedium" or name == "ForestLarge":
        """
            Actions:
            There are 6 discrete deterministic actions:
            - 0: wait
            - 1: cut
        """
        return ["W","C"]
    return []


if __name__ == "__main__":
    print("#####################################################################")
    print("CS7641 ML - Reinforcement Learning Assignment Test Program")
    print("#####################################################################")
    run_mdp_forest(run_value_iteration=True, run_policy_iteration=True, run_q_learning=True)
    run_mdp_gym(run_value_iteration=True, run_policy_iteration=True, run_q_learning=True)

