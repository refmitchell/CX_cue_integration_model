"""
snapshot_preservation.py

Generate an initial snapshot, then modify the cue configuration and
attempt to learn over the existing snapshot to see how long it takes
to update. This module will generate the snapshots and plots.
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
    change = 24
    for t in range(500):
        c1+=change
        c2+=change
        rm.update_state(c1,
                        c2,
                        sm=change,
                        w1=w1,
                        w2=w2,
                        plasticity=False
        )

    return rm

def snapshot_preservation():
    """
    Learn a mapping with one cue strong, the next week
    """
    duration = 15 # Simulation duration - degrees
    n_r = 8
    base = 1/n_r
    iterations = 4

    r1_snapshot = np.zeros((iterations, n_r, n_r))
    r2_snapshot = np.zeros((iterations, n_r, n_r))

    """
    Initial, 1 strong, 2 weak.
    """
    rm = RingModel({rmkeys.n_r:n_r})
    rm = initialise(rm, c1=0, c2=0)

    # 'zero' the adjacency matrices for learning (ONLY ON FIRST DANCE)
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
                        w1=0.8,
                        w2=0.2,
                        sm=angle_update,
                        plasticity=True
        )

    r1_snapshot[0] = rm.w_r1_epg
    r2_snapshot[0] = rm.w_r2_epg

    for it in range(iterations-1):
        idx = it + 1

        # Move cues apart, conflict plus strength change
        # Demonstrates that the old mappings affect the current
        # ones, even if the mapping shifts relatively cleanly.
        c1=0
        c2=179

        for t in range(duration):
            angle_update = 24 # 6deg per timestep

            c1+=angle_update
            c2+=angle_update

            # Learning bout
            rm.update_state(c1,
                            c2,
                            w1=0.2,
                            w2=0.8,
                            sm=angle_update,
                            plasticity=True
            )

        r1_snapshot[idx] = rm.w_r1_epg
        r2_snapshot[idx] = rm.w_r2_epg


    fig, axs = plt.subplots(nrows=iterations, ncols=2, sharex=True, sharey=True)
    fig.set_size_inches((4,8))
    fig.tight_layout()
    vmin = 0
    vmax = 0.25
    cmap = 'Greys_r'
    for idx in range(axs.shape[0]):
        curr_ax = axs[idx]
        wmap = curr_ax[0].pcolormesh(r1_snapshot[idx], vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
        wmap.set_edgecolor('face')
        wmap = curr_ax[1].pcolormesh(r2_snapshot[idx], vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
        wmap.set_edgecolor('face')

    subtitles = [
        "a) Initial snapshot, Cue 1 strong",
        "b) 1st iteration, Cue 2 strong",
        "c) 2nd iteration",
        "d) 3rd iteration"
        ]
    xidx = 0
    yidx = 0
    for ax_row in axs:
        for plot in ax_row:
#            plot.set_aspect("equal")
            plt.subplots_adjust(wspace=0.08, hspace=None)
            plot.set_ylim([8,0])
            plot.set_xlim([0,n_r])
            if xidx == 0: # Only for left-hand plots
                plot.set_ylabel("E-PG index")
                plot.text(-0.2, -0.5,subtitles[yidx], ha='left')

            if yidx == 3:
                plot.set_xlabel("R index")
                if n_r==8:
                    plot.set_xticks([0, 2, 4, 6, 8])
                else:
                    plot.set_xticks(np.arange(0,n_r,1))
            xidx += 1

        xidx = 0
        yidx +=1

    plt.savefig("plots/snapshot_preservation_aligned.pdf", dpi=300,bbox_inches="tight")

if __name__ == "__main__":
    snapshot_preservation()
