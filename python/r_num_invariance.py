"""
r_num_invariance.py

Here we tested whether using more or less R neurons significantly
affected the vector sum function of the network. We tested the
minimum of 3 R neurons up to 24.
"""

import numpy as np
import matplotlib.pyplot as plt
from extended_ring_model import *
from dict_key_definitions import rmkeys, decodekeys
from test_utilities import *

def initialise(rm, c1=0, c2=0, w1=0.5, w2=0.5):
    """
    Initialisation routine
    :param rm: a RingModel to initialise before learning
    :return: the initialised RingModel
    """
    rm.reset_rates()
    for t in range(500):
        rm.update_state(c1,
                        c2,
                        sm=24,
                        w1=w1,
                        w2=w2,
                        plasticity=False
        )

    return rm

def learning_and_conflict(n_r, n, weights, conflicts):
    duration = 15
    base = 1/n_r
    rm = RingModel({rmkeys.n_r:n_r})
    rm = initialise(rm, c1=0, c2=0)

    # 'zero' the adjacency matrices for learning
    rm.w_r1_epg = np.zeros((8,n_r)) + base
    rm.w_r2_epg = np.zeros((8,n_r)) + base

    c1=0
    c2=0
    for t in range(duration):
        angle_update = 24 # 6deg per timestep

        c1+=angle_update
        c2+=angle_update

        rm.update_state(c1,
                        c2,
                        sm=angle_update,
                        plasticity=True
        )

    n_weights = rm.w_r1_epg
    n_conflicts = np.zeros((len(weights), n))

    prm_offset = 0

    for idx in range(len(weights)):
        rm.reset_rates()
        w1 = weights[idx]
        w2 = 1 - w1
        ring_out = []
        for cue_two in conflicts:
            rm.update_state(0,
                            cue_two,
                            sm=0,
                            w1=w1,
                            w2=w2,
                            plasticity=False)

            static_t, static_r = rm.decode()[decodekeys.epg]
            ring_out.append(np.rad2deg(static_t) % 360)

        ring_offset = ring_out[0]
        ring_out = [x - ring_offset for x in ring_out]

        n_conflicts[idx] = [x % 360 for x in ring_out]

    return (n_weights, n_conflicts)

def snapshot_comparison():
    """
    Create 4x2 figure showing conflict behaviour and snapshots for
    different numbers of R neurons.
    """
    n = 180
    conflicts = np.linspace(0,180,n)
    weights = np.arange(0.1, 1, 0.1)
    duration = 15 # Learning duration

    n3 = learning_and_conflict(3, n, weights, conflicts)
    n6 = learning_and_conflict(6, n, weights, conflicts)
    n12 = learning_and_conflict(12, n, weights, conflicts)
    n24 = learning_and_conflict(24, n, weights, conflicts)

    data_full = [n3, n6, n12, n24]

    """
    Plotting
    """
    fig1 = plt.figure(figsize=(7,7))
    fig2 = plt.figure(figsize=(7,7))
    subplots = list(fig1.subplots(nrows=2,ncols=2))
    sp2 = list(fig2.subplots(nrows=2,ncols=2))
    subplots.append(sp2[0])
    subplots.append(sp2[1])
    print(subplots)

    # Conflict data
    inc = 60
    ticks = np.arange(0,190+inc,inc)
    for idx in range(len(subplots)):
        conf_axs = subplots[idx][1]
        conf_data = data_full[idx][1]
        for jdx in range(len(conf_data)):
            condition = conf_data[jdx]
            w1 = weights[jdx]
            label = "w1 = {:.1f}".format(w1) if jdx == 0 else "{:.1f}".format(w1)
            conf_axs.plot(condition, label=label)

        conf_axs.set_xticks([0,60,120,180],
                            labels=["0$^\degree$",
                                    "60$^\degree$",
                                    "120$^\degree$",
                                    "180$^\degree$"])
        conf_axs.set_yticks([0,60,120,180],
                            labels=["0$^\degree$",
                                    "60$^\degree$",
                                    "120$^\degree$",
                                    "180$^\degree$"])

        conf_axs.set_xlim([0,180])
        conf_axs.set_ylim([0,180])
        conf_axs.set_ylabel("Integrated angle")
        conf_axs.set_aspect("equal")
        if (idx%2)!=0:
            conf_axs.set_xlabel("Cue conflict")
        else:
            fig = fig1 if idx==0 else fig2
            conf_axs.legend(loc="center",
                            bbox_to_anchor=(0.5, 0),
                            bbox_transform=fig.transFigure,
                            ncols=len(weights),
                            prop={"size":8}
            )


    subtitles = [
        "a) n = 3",
        "b) n = 6",
        "a) n = 12",
        "b) n = 24"
        ]

    ns = [3, 6, 12, 24]
    # Plot data
    cmap = 'Greys_r'
    vmax = 0.25

    for idx in range(len(subplots)):
        snap_axs = subplots[idx][0]
        snap_data = data_full[idx][0]
        n_r = ns[idx]
        step = 3
        ticks = np.arange(0, n_r+step, step)

        wmap = snap_axs.pcolormesh(snap_data, vmin=0, vmax=vmax, cmap=cmap)
        wmap.set_edgecolor('face')

        fig1.subplots_adjust(wspace=0.3, hspace=0.3)
        fig2.subplots_adjust(wspace=0.3, hspace=0.3)
        snap_axs.set_xticks(ticks)
        snap_axs.set_ylim([8,0])
        snap_axs.set_xlim([0,n_r])
        snap_axs.set_ylabel("E-PG index")
        snap_axs.text(0, -0.3,subtitles[idx], ha='left')

        if (idx%2)!=0:
            snap_axs.set_xlabel("R index")

    fig1.savefig("plots/r_invariance_1.pdf", bbox_inches="tight")
    fig2.savefig("plots/r_invariance_2.pdf", bbox_inches="tight")
#    plt.show()

if __name__ == "__main__":
    snapshot_comparison()
