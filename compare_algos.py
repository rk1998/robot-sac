import numpy as np
import matplotlib.pyplot as plt
import sys



def plot_reward_results(algo_1_arr, algo_2_arr, algo_1_name, algo_2_name, title="Comparison"):
    plt.figure()
    plt.plot(algo_1_arr, "-b", label=algo_1_name)
    plt.plot(algo_2_arr, "-r", label=algo_2_name)
    figure_title = algo_1_name + " vs " + algo_2_name + " average rewards during training"
    plt.title(figure_title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()


def main():
    if len(sys.argv) < 5:
        print("Usage: python compare_algos.py <algo 1 reward file> <algo 2 reward file> <algo 1 name> <algo 2 name>")
        exit()

    algo_1_file = sys.argv[1]
    algo_2_file = sys.argv[2]
    algo_1_name = sys.argv[3]
    algo_2_name = sys.argv[4]
    algo_1_arr = np.load(algo_1_file)
    algo_2_arr = np.load(algo_2_file)
    plot_reward_results(algo_1_arr, algo_2_arr, algo_1_name, algo_2_name)

if __name__ == '__main__':
    main()