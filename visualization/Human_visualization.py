from cProfile import run
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import random
from matplotlib import ticker
num_runs = 1
one_row = True

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

n_moving = 2

names = [
    "output/NODE/ode-rnn_NODE.csv",
    "output/GHBNODE/ode-rnn_GHBNODE.csv",
    "output/GNesterovNODE/ode-rnn_GNesterovNODE.csv",
    "output/GRMSpropNODE/ode-rnn_GRMSpropNODE.csv",
    "output/GTOAESNODE/ode-rnn_GTOAESNODE.csv",
]


alt_names = ["NODE-RNN", 
             "GHBNODE-RNN",
             "GNesterovNODE-RNN",
             "GRMSpropNODE-RNN",
             "GTOAES-NODE-RNN",
]

df_names = {}
for j, name in enumerate(names):
    filepath = name
    df = pd.read_csv(filepath, header=None, names=["epoch", 
							"test_loss", 
							"test_likelihood", 
							"test_kl_first_p", 
							"test_std_first_p", 
							"train_kl_coef",
							"train_loss",
							"train_ce_loss",
							"train_nfe",
							"train_nbe",
       						"test_nfe",
                            "train_time",
                            "test_time",
                            "test_mse",
                            "test_acc",
                            "test_pois_likelihood",
                            "test_ce_loss",])
    # df = df.head(50)
    print(alt_names[j], " (train nfe mean)", np.mean(df["train_nfe"]))
    print(alt_names[j], " (train nbe mean)", np.mean(df["train_nbe"]))
    print(alt_names[j], " (test acc max)", np.max(df["test_acc"]))
    if alt_names[j] not in df_names:
        df_names[alt_names[j]] = []
    df_names[alt_names[j]].append(df)


colors = [
	"mediumvioletred",
	"navy",
 	"darkorange",
	"darkorchid",
    "mediumseagreen"
]

line_styles = [
	':',
	'-.',
	'-',
	'-',
	'-'
]

line_widths = [
    5,
    5,
    7,
    7,
    7
]


if one_row:
    font = {'size'   : 40}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(19, 7))
    gs = fig.add_gridspec(1, 2, hspace=0.2, wspace=0.5)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    axes = (ax1, ax2)
else:
    font = {'size'   : 40}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(30, 15))
    gs = fig.add_gridspec(2, 6, hspace=0.30, wspace=2.5)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:])
    ax4 = fig.add_subplot(gs[1, 1:3])
    ax5 = fig.add_subplot(gs[1, 3:5])
    axes = (ax1, ax2, ax4)

alt_attr_names = ["Forward NFEs", "Backward NFEs"]
attr_names = ["train_nbe", "train_nfe"]
for j, attribute in enumerate(attr_names):
    for i, name in enumerate(alt_names): 
        df_name = df_names[name][0]
        attr_arr = df_name[attribute]
        attr_arr = moving_average(attr_arr, n_moving)
        epoch_arr = df_name["epoch"]
        axes[j].plot(epoch_arr, attr_arr, line_styles[i], linewidth=line_widths[i], color=colors[i], label=alt_names[i])
    if attribute == "train_nfe":
        axes[j].set_ylim((170, 310))
    axes[j].set_xlim((2, 50))
    axes[j].set(xlabel="Epoch", ylabel=f"{alt_attr_names[j]}")
    axes[j].grid()

axbox = axes[-1].get_position()
if one_row:
	_ = plt.legend(bbox_to_anchor=(0.5, axbox.y0-0.5), loc="lower center", 
					bbox_transform=fig.transFigure, ncol=3, handletextpad=0.5, columnspacing=0.6, borderpad=0.3)
else:
	_ = plt.legend(bbox_to_anchor=(0.5, axbox.y0-0.225), loc="lower center", 
					bbox_transform=fig.transFigure, ncol=4, handletextpad=0.5, columnspacing=0.6, borderpad=0.3)
filepath = f"human_viz.pdf"
print("Saving to", filepath)
plt.savefig(filepath, transparent = True, bbox_inches = 'tight', pad_inches = 0)
plt.show()
