"""
noise_filtering.py

Observing snapshots over repeated dances during behavioural simulations
raised the question of whether repeated dances could help filter out noise.
From the behavioural simulations it seemed that they did. On closer inspection,
this appeared to be an inconsistent phenomenon which could lead to a clean
snapshot or could lead to a corrupted snapshot. There is a clear dependence
on the concentration of the von Mises noise experienced.

This links back to the weighting mechanics (conflict_weight.py) simulations.
Unreliable cues may be weighted less or they may not be, there is no reliable
weighting effect from reliability alone. This helps to explain the huge amount
of variability experienced in those simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from extended_ring_model import *
from dict_key_definitions import rmkeys, decodekeys
from test_utilities import *

def snapshot_preservation():
    """
    Learn a mapping with one cue strong, the next weak
    """
    duration = 15 # Simulation duration - degrees
    n_r = 8
    base = 1/n_r
    iterations = 4

    # 'Good' snapshots
    gr1_snapshot = np.zeros((iterations, n_r, n_r))
    gr2_snapshot = np.zeros((iterations, n_r, n_r))

    # 'Bad' snapshots
    br1_snapshot = np.zeros((iterations, n_r, n_r))
    br2_snapshot = np.zeros((iterations, n_r, n_r))

    # Init models
    grm =  RingModel({rmkeys.n_r:n_r})
    brm =  RingModel({rmkeys.n_r:n_r})
    grm.initialise()
    brm.initialise()

    # Flatten weights for first dance
    grm.w_r1_epg = np.zeros((8,n_r)) + base
    grm.w_r2_epg = np.zeros((8,n_r)) + base
    brm.w_r1_epg = np.zeros((8,n_r)) + base
    brm.w_r2_epg = np.zeros((8,n_r)) + base

    # Cues always have the same 'strength'
    w1=0.5
    w2=0.5

    s1 = np.random.randint(0, high=100000000)
    s2 = np.random.randint(0, high=100000000)

    # 'Good' seeds (the noise experienced leads to clean snapshots)
    gs1 = 73749398 # Comment for random seeds
    gs2 = 11516574 # Comment for random seeds

    # 'Bad' seeds (the noise experienced leads to a corrupted snapshot)
    bs1 = 3987523
    bs2 = 53045709

    print("G1: {}".format(gs1))
    print("G2: {}".format(gs2))
    print("")
    print("B1: {}".format(bs1))
    print("B2: {}".format(bs2))

    k1=2
    k2=k1/4

    # Seed random states
    ggen1 = np.random.RandomState(gs1)
    ggen2 = np.random.RandomState(gs2)
    bgen1 = np.random.RandomState(bs1)
    bgen2 = np.random.RandomState(bs2)

    c1 = 0
    c2 = 0
    for it in range(iterations):
        for t in range(duration):
            angle_update = 24

            c1 += angle_update
            c2 += angle_update

            # Angles experienced are the same, just the noise changes.
            gc1_sample = c1 + np.degrees(ggen1.vonmises(0, k1))
            gc2_sample = c2 + np.degrees(ggen2.vonmises(0, k2))

            bc1_sample = c1 + np.degrees(bgen1.vonmises(0, k1))
            bc2_sample = c2 + np.degrees(bgen2.vonmises(0, k2))

            grm.update_state(gc1_sample,
                             gc2_sample,
                             sm=angle_update,
                             plasticity=True
            )

            brm.update_state(bc1_sample,
                             bc2_sample,
                             sm=angle_update,
                             plasticity=True
            )

        gr1_snapshot[it] = grm.w_r1_epg
        gr2_snapshot[it] = grm.w_r2_epg

        br1_snapshot[it] = brm.w_r1_epg
        br2_snapshot[it] = brm.w_r2_epg

    g_snapshot = [gr1_snapshot, gr2_snapshot]
    b_snapshot = [br1_snapshot, br2_snapshot]

    # The noise 'filtering' seems to be quite variable so we compare a 'good'
    # example where noise is suitably filtered and a 'bad' example
    mosaic = [
        ["g11", "g12", "b11", "b12"],
        ["g21", "g22", "b21", "b22"],
        ["g31", "g32", "b31", "b32"],
        ["g41", "g42", "b41", "b42"]
    ]

    fig, axs = plt.subplot_mosaic(mosaic, sharex=True, sharey=True)
    fig.set_size_inches((8,8))
    fig.tight_layout()

    # Plot weight matrices
    vmin = 0
    vmax = 0.25
    cmap = 'Greys_r'
    for yidx in range(len(mosaic)):
        row_idx = yidx + 1
        for col_idx in [1,2]:
            grc = "g{}{}".format(row_idx, col_idx)
            brc = "b{}{}".format(row_idx, col_idx)

            wmap = axs[grc].pcolormesh(g_snapshot[col_idx - 1][yidx],
                                       vmin=vmin, vmax=vmax, rasterized=True, cmap=cmap)
            wmap.set_edgecolor('face')
            wmap = axs[brc].pcolormesh(b_snapshot[col_idx - 1][yidx],
                                       vmin=vmin, vmax=vmax, rasterized=True, cmap=cmap)
            wmap.set_edgecolor('face')
            axs[grc].set_ylim([8,0])
            axs[grc].set_xlim([0,n_r])
            axs[grc].set_aspect('equal')
            axs[brc].set_ylim([8,0])
            axs[brc].set_xlim([0,n_r])
            axs[brc].set_aspect('equal')

    g_subtitles = [
        "Ai) \"Good\" noise, 1st iteration",
        "Bi) 2nd iteration",
        "Ci) 3rd iteration",
        "Di) 4th iteration"
        ]

    b_subtitles = [
        "Aii) \"Bad\" noise, 1st iteration",
        "Bii) 2nd iteration",
        "Cii) 3rd iteration",
        "Dii) 4th iteration"
        ]

    # Add y-axis labels to the left-hand plots
    for row in range(iterations):
        lkey = "g{}1".format(row + 1)
        rkey = "b{}1".format(row + 1)
        axs[lkey].set_ylabel("E-PG index")
        axs[lkey].text(-0.2, -0.5, g_subtitles[row], ha='left')
        axs[rkey].text(-0.2, -0.5, b_subtitles[row], ha='left')


    # Add x-axis labels to the bottom plots
    for col in [1,2]:
        for lab in ["g", "b"]:
            key = "{}4{}".format(lab, col)
            axs[key].set_xticks([0, 2, 4, 6, 8])
            axs[key].set_xlabel("R index")


    plt.savefig("plots/noise_filter.svg", dpi=300,bbox_inches="tight")
    plt.savefig("plots/noise_filter.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    snapshot_preservation()
