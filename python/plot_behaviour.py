"""
plot_behaviour.py

Plot the behavioural data contained in the behaivoural_data sub-directory.
This script generates a companion plot for plot_precision.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon # Statistical comparison between groups

import simulation_utilities as simutils
import dict_key_definitions as keydefs
from dict_key_definitions import statskeys, simkeys
from dict_key_definitions import precision_reskeys as reskeys
from test_utilities import circmean

reskeys = keydefs.constkeys_precision_experiment_result_dict()
statskeys = keydefs.constkeys_summary_stats_dict()
if __name__ == "__main__":
    plt.rcParams["svg.fonttype"] = 'none' # Embed text objects in SVGs

    # Load stats data from file
    stats = dict()
    stats["45"] = pd.read_csv("behavioural_data/rvals-45.csv")
    stats["75"] = pd.read_csv("behavioural_data/rvals-75.csv")


    fig = plt.figure("Precision experiments", figsize=(12,5))
    fig.suptitle("(Behaviour) Changes in precision when adding or removing a cue")

    mosaic = [["45-s-sw", "45-w-sw", "45-sw-s", "45-sw-w"],
              ["75-s-sw", "75-w-sw", "75-sw-s", "75-sw-w"]
    ]
    axs = fig.subplot_mosaic(mosaic,
                             sharey=True,
                             sharex=True)

    titles = dict()
    titles["45-s-sw"] = "(i) add wind"
    titles["45-w-sw"] = "(ii) add sun"
    titles["45-sw-s"] = "(iii) remove wind"
    titles["45-sw-w"] = "(iv) remove sun"
    titles["75-s-sw"] = "(v) add wind"
    titles["75-w-sw"] = "(vi) add sun"
    titles["75-sw-s"] = "(vii) remove wind"
    titles["75-sw-w"] = "(viii) remove sun"

    # Generate a boxplot for each experiment
    bg_artists = []
    bg_labels = []
    box_artists = []
    box_labels = []
    for jdx in range(len(axs.keys())):
        k = list(axs.keys())[jdx] # axis key

        components = k.split("-")
        elevation = components[0] # Key for stats dict

        # Construct column key for each condition
        iek = "-".join(components[1:]) + "-0"
        tek = "-".join(components[1:]) + "-1"

        # Extract dataand filter nan entries
        imvls = stats[elevation][iek]
        tmvls = stats[elevation][tek]
        initial_mean_vector_lengths = imvls[~np.isnan(imvls)]
        test_mean_vector_lengths = tmvls[~np.isnan(tmvls)]

        # Replicate biological stats analysis (Wilcoxon signed-rank or paired
        # Wilcoxon test).
        stat, p_value = wilcoxon(initial_mean_vector_lengths,
                                 test_mean_vector_lengths)

        boxcolour='k'
        blues = [reskeys.exp1, reskeys.exp2, reskeys.exp5, reskeys.exp6]
        wbr = [reskeys.exp1, reskeys.exp2, reskeys.exp3, reskeys.exp4]
        facecol = "#9EB6D1" if len(components[1]) < 2 else "#FFECC0"
        bgcolour = 'tab:purple' if elevation == "45" else 'tab:cyan'
        bglabel = None

        if k == "45-s-sw":
            bglabel = r"$45^\degree$ elevation"
        elif k == "75-s-sw":
            bglabel = r"$75^\degree$ elevation"

        bg = axs[k].fill_between(np.linspace(0,3), 0, 1,
                                 color=bgcolour, alpha=0.15, zorder=0)
        if k == "45-s-sw" or k == "75-s-sw":
            bg_labels.append(bglabel)
            bg_artists.append(bg)

        boxlabel = None
        if k == "45-s-sw":
            boxlabel = "Add cue"
        elif k == "45-sw-s":
            boxlabel = "Remove cue"

        box_dict = axs[k].boxplot([initial_mean_vector_lengths, test_mean_vector_lengths],
                                  showfliers=True,
                                  sym=".",
                                  patch_artist=True,
                                  boxprops=dict(facecolor=facecol, color=boxcolour),
                                  capprops=dict(color=boxcolour),
                                  whiskerprops=dict(color=boxcolour),
                                  medianprops=dict(color=boxcolour),
                                  zorder=1,
                                  widths=0.5
        )

        p_string = "n.s."
        if p_value <= 0.05: # If minimally signficant, iterate.
            sig_levels = [0.05, 0.01, 0.001, 0.0001, 0.00001]
            max_significance = max(sig_levels)
            for s in sig_levels:
                if p_value < s:
                    max_significance = s

            p_string = "$p < {}$".format(max_significance)

        axs[k].text(2.9, 0.03, p_string,ha="right")

        if k == "45-s-sw" or k == "45-sw-s":
            box_labels.append(boxlabel)
            box_artists.append(box_dict["boxes"][0])

        axs[k].set_ylim([0,1])
        axs[k].set_xlim([0,3])
        axs[k].set_title(titles[k])
        axs[k].set_aspect(2.8)

        if jdx >= 4:
            axs[k].set_xticks([1,2])
            axs[k].set_xticklabels(["initial", "test"])


    axs["45-s-sw"].set_ylabel("Mean vector length")
    axs["75-s-sw"].set_ylabel("Mean vector length")

    labels = bg_labels + box_labels
    artists = bg_artists + box_artists
    fig.legend(bg_artists,
               bg_labels,
               ncols=8,
               prop={'size':8},
               loc=10,
               bbox_to_anchor=(0.5, 0),
               bbox_transform=fig.transFigure)

    fig.legend(box_artists,
               box_labels,
               ncols=8,
               prop={'size':8},
               loc=10,
               bbox_to_anchor=(0.5, -0.06),
               bbox_transform=fig.transFigure)

    plt.savefig("plots/behaviour.svg", bbox_inches="tight")

