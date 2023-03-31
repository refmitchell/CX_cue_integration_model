"""
plot_modality_transfer.py

Load and plot data generated by modality_transfer.py.
"""
import json
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd

import dict_key_definitions as keydefs
from dict_key_definitions import statskeys, simkeys
from dict_key_definitions import modality_transfer_reskeys as reskeys

from test_utilities import circ_scatter, confidence_interval, angular_deviation, circmean, v_test

if __name__ == "__main__":
    # Load stats data from file
    with open("data_out/modality_transfer_stats.json") as f:
        stats = json.load(f)

    # Load simulation metadata
    with open("data_out/modality_transfer_meta.json") as f:
        meta = json.load(f)

    # Dacke et al. (2019) show the change in bearing between rolls 1 and 4
    # for each individual. Extract exit angles for each individual for roll 1.
    # Then extract exit angles for each individual for roll 4. Round to nearest
    # five degrees and compute the difference.

    # Exit angles, should be one entry for each agent (n = 40).
    # Each agent has a list of exits so these should be flattened.
    roll_one_exits = stats[reskeys.roll1][statskeys.exit_angles]
    roll_one_exits = list(itertools.chain(*roll_one_exits))
    roll_one_exits = np.radians(
        np.around(np.degrees(roll_one_exits)/5, decimals=0)*5
    )

    roll_four_exits = stats[reskeys.roll4][statskeys.exit_angles]
    roll_four_exits = list(itertools.chain(*roll_four_exits))
    roll_four_exits = np.radians(
        np.around(np.degrees(roll_four_exits)/5, decimals=0)*5
    )

    changes = roll_one_exits - roll_four_exits

    # Load in Dacke et al. 2019 data
    dacke_changes = pd.read_csv(
        "dacke2019_data/mt_changes_dacke2019.csv"
    )["change"].to_numpy()
    dacke_changes = np.radians(dacke_changes)

    dacke_mean = circmean(dacke_changes)
    mean = circmean(changes)

    dacke_v_pval, dacke_v = v_test(dacke_changes, 0)
    v_pval, v = v_test(changes, 0)
    print("Model - V Test: V = {:.2f}, p = {}".format(v, v_pval))
    print("Dacke - V Test: V = {:.2f}, p = {}".format(dacke_v, dacke_v_pval))

    radial_base = 1
    # Compute radii for stacked circular scatter plot.
    radii, angle_out = circ_scatter(changes,
                                    radial_interval=-0.1,
                                    radial_base=radial_base)

    # Batschelet (1981), Eq. (2.3.2)
    ci95 = confidence_interval(mean[0], 0.05, 40)
    angular_std_dev = angular_deviation(mean[0])
    half_arc = ci95/2
    csd_arc = np.linspace(mean[1] - half_arc, mean[1] + half_arc, 100)
    mean_line_rs = np.linspace(0, radial_base + 0.1, 100)

    fig, axs = plt.subplot_mosaic([["model","dacke"]],
                                  figsize=(8,3),
                                  subplot_kw={"projection":"polar"})


    axs["model"].set_title("Neural model",fontsize=14)
    axs["model"].scatter(angle_out, radii, edgecolors='k',s=50)
    axs["model"].plot(np.zeros(len(mean_line_rs)) + mean[1],
                      mean_line_rs,
                      color='k',
                      zorder=0)
    axs["model"].text(0,
                      0,
                      "$\mu$ = {:.2f}$^\degree$\n$s0$ = {:.2f}$^\degree$\n$V = {:.2f}$\n$p = {:.7f}$"
                      .format(np.degrees(mean[1]), np.degrees(angular_std_dev), v, v_pval),
                      ha="center",
                      va="center",
                      bbox=dict(facecolor='1', edgecolor='k', pad=1.5)
    )

    axs["model"].plot(csd_arc, np.zeros(len(csd_arc)) + radial_base + 0.1, color='k')

    # Plot Dacke data and stats
    radii, angle_out = circ_scatter(dacke_changes,
                                    radial_interval=-0.1,
                                    radial_base=radial_base)
    ci95 = confidence_interval(dacke_mean[0], 0.05, 40)
    angular_std_dev = angular_deviation(dacke_mean[0])
    half_arc = ci95/2
    csd_arc = np.linspace(mean[1] - half_arc, mean[1] + half_arc, 100)
    mean_line_rs = np.linspace(0, radial_base + 0.1, 100)

    axs["dacke"].set_title("Dacke et al. (2019)",fontsize=14)
    axs["dacke"].scatter(angle_out, radii, edgecolors='k',s=50)
    axs["dacke"].plot(np.zeros(len(mean_line_rs)) + dacke_mean[1],
                      mean_line_rs,
                      color='k',
                      zorder=0)
    axs["dacke"].text(0,
                      0,
                      "$\mu$ = {:.2f}$^\degree$\n$s0$ = {:.2f}$^\degree$\n$V = {:.2f}$\n$p = {:.7f}$"
                      .format(np.degrees(dacke_mean[1]), np.degrees(angular_std_dev), dacke_v, dacke_v_pval),
                      ha="center",
                      va="center",
                      bbox=dict(facecolor='1', edgecolor='k', pad=1.5)
    )

    axs["dacke"].plot(csd_arc, np.zeros(len(csd_arc)) + radial_base + 0.1, color='k')    
    for k in axs.keys():
        axs[k].set_theta_direction(-1)
        axs[k].set_theta_zero_location("N")
        axs[k].set_xticks([0,np.pi/2, np.pi, 3*np.pi/2],
                             labels=["0$^\degree$",
                                     "90$^\degree$",
                                     "180$^\degree$",
                                     "-90$^\degree$"],
                             fontsize=14)
        axs[k].set_ylim([0,radial_base + 0.15])
        axs[k].set_yticks([])


    fig.savefig("plots/modality_transfer.pdf", bbox_inches="tight")

