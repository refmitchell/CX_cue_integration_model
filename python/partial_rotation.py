"""
partial_rotation.py

Perform partial learning rotations and plot adjacency
matrices and conflict outputs. The goal was to determine
if partial rotations were still useful which it seems they are.
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

def learning_and_conflict(duration, n, weights, conflicts):
    duration = duration
    n_r = 8
    base = 1/n_r
    rm = RingModel({rmkeys.n_r:n_r})
    rm.initialise()

    # 'zero' the adjacency matrices for learning
    rm.w_r1_epg = np.zeros((8,n_r)) + base
    rm.w_r2_epg = np.zeros((8,n_r)) + base

    c1=0
    c2=0
    for t in range(duration):
        angle_update = 24

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

    n3 = learning_and_conflict(0, n, weights, conflicts)
    n6 = learning_and_conflict(5, n, weights, conflicts)
    n12 = learning_and_conflict(10, n, weights, conflicts)
    n24 = learning_and_conflict(15, n, weights, conflicts)

    data_full = [n3, n6, n12, n24]

    """
    Plotting
    """
#     fig1 = plt.figure(figsize=(7,7))
#     fig2 = plt.figure(figsize=(7,7))
#     subplots = list(fig1.subplots(nrows=2,ncols=2))
#     sp2 = list(fig2.subplots(nrows=2,ncols=2))
#     subplots.append(sp2[0])
#     subplots.append(sp2[1])
#     print(subplots)

#     # Conflict data
#     inc = 60
#     ticks = np.arange(0,190+inc,inc)
#     for idx in range(len(subplots)):
#         conf_axs = subplots[idx][1]
#         conf_data = data_full[idx][1]
#         for jdx in range(len(conf_data)):
#             condition = conf_data[jdx]
#             w1 = weights[jdx]
#             conf_axs.plot(condition, label="w1 = {:.1}".format(w1))

#         conf_axs.set_xticks(ticks)
#         conf_axs.set_yticks(ticks)
#         conf_axs.set_xlim([0,180])
#         conf_axs.set_ylim([0,180])
#         conf_axs.set_ylabel("Integrated angle (degrees)")
#         conf_axs.set_aspect("equal")
#         if (idx%2)!=0:
#             conf_axs.set_xlabel("Cue conflict (degrees)")
#         else:
#             conf_axs.legend(fontsize="small", loc="center", bbox_to_anchor=(0.22, 0.75))

#     subtitles = [
#         "a) No rotation",
#         "b) 1/3rd rotation",
#         "a) 2/3rd rotation",
#         "b) Full rotation"
#         ]

#     ns = [3, 6, 12, 24]
#     # Plot data
#     cmap = 'hot'#'binary'#'hot'
#     vmax = 1
# #    vmax = 0.5

#     for idx in range(len(subplots)):
#         snap_axs = subplots[idx][0]
#         snap_data = data_full[idx][0]
#         n_r = 8
#         step = 2
#         ticks = np.arange(0, n_r+step, step)

#         wmap = snap_axs.pcolormesh(snap_data, vmin=0, vmax=vmax, cmap=cmap)
#         wmap.set_edgecolor('face')

#         fig1.subplots_adjust(wspace=0.3, hspace=0.3)
#         fig2.subplots_adjust(wspace=0.3, hspace=0.3)
#         snap_axs.set_xticks(ticks)
#         snap_axs.set_ylim([8,0])
#         snap_axs.set_xlim([0,8])
#         snap_axs.set_ylabel("E-PG index")
#         snap_axs.text(0, -0.3,subtitles[idx], ha='left')

#         if (idx%2)!=0:
#             snap_axs.set_xlabel("R index")

#     fig1.savefig("partial_1.pdf", bbox_inches="tight")
#     fig2.savefig("partial_2.pdf", bbox_inches="tight")

    nrows = 2
    ncols = len(data_full)
    poster_fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8,4))
    plt.tight_layout()
    ss_axes = axs[0]
    conf_axes = axs[1]

    ns = [3, 6, 12, 24]
    subtitles = [
        "a) No rotation",
        "b) 1/3rd rotation",
        "a) 2/3rd rotation",
        "b) Full rotation"
    ]


    for idx in range(ncols):
        # Unpack data
        data = data_full[idx]
        ss_data = data[0]
        conf_data = data[1]
        ss_ax = ss_axes[idx]
        conf_ax = conf_axes[idx]

        # Plot snapshot
        wmap = ss_ax.pcolormesh(ss_data, vmin=0, vmax=0.25, cmap='Greys_r')
        wmap.set_edgecolor('face')

        n_r = 8
        step = 2
        ticks = np.arange(0, n_r+step, step)
        ss_ax.set_xticks(ticks)
        ss_ax.set_ylim([8,0])
        ss_ax.set_xlim([0,8])
        ss_ax.set_yticks(ticks)
        ss_ax.set_xlabel("R index")
        if idx == 0:
            ss_ax.set_ylabel("E-PG index")
        else:
            ss_ax.tick_params(labelleft=False)

        ss_ax.text(0, -0.3,subtitles[idx], ha='left')
        ss_ax.set_aspect('equal')

        # Plot conflicts
        for jdx in range(len(conf_data)):
            condition = conf_data[jdx]
            w1 = weights[jdx]
            label = "{:.1f}".format(w1)
            if jdx == 0:
                label = "w1  = {:.1f}".format(w1)

            conf_ax.plot(condition, label=label)

        # Format plots
        inc = 60
        ticks = np.arange(0,190+inc,inc)
        labels = ["{}$^\degree$".format(x) for x in ticks]
        conf_ax.set_xticks(ticks, labels=labels)
        conf_ax.set_yticks(ticks, labels=labels)
        conf_ax.set_xlabel("Cue conflict")
        if idx == 0:
            conf_ax.set_ylabel("Integrated angle")
            conf_ax.legend(loc='center',
                           ncols=len(weights),
                           bbox_to_anchor=(0.5, -0.05),
                           bbox_transform=poster_fig.transFigure,
                           prop={"size":8})
        else:
            conf_ax.tick_params(labelleft=False)

        conf_ax.set_xlim([0,180])
        conf_ax.set_ylim([0,180])
        conf_ax.set_aspect("equal")

    poster_fig.subplots_adjust(wspace=0, hspace=0.5)
    poster_fig.savefig("plots/partial_rotations.pdf", bbox_inches='tight')
#    plt.show()



if __name__ == "__main__":
    snapshot_comparison()
