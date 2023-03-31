"""
snapshot_comparison.py

This module performs learning rotations with different cue
conditions to determine effects on the R to E-PG mapping.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from extended_ring_model import *
from dict_key_definitions import rmkeys, decodekeys
from test_utilities import *

def initialise(rm, c1=0, c2=0, w1=0.5, w2=0.5, duration=500):
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

def snapshot_comparison():
    """
    Create 4x2 figure showing different snapshot scenarios:
    - Noisy cues
    - Cues in large conflict
    - One reliable, one not
    - One strong, one weak
    """
    duration = 15 # Simulation duration - degrees
    n_r = 8
    base = 1/n_r

    """
    Noisy
    """
    rm = RingModel({rmkeys.n_r:n_r})
    rm = initialise(rm, c1=0, c2=0)

    # 'zero' the adjacency matrices for learning
    rm.w_r1_epg = np.zeros((8,n_r)) + base
    rm.w_r2_epg = np.zeros((8,n_r)) + base
    print(rm.w_r1_epg.shape)
    print(rm.epg_rates.shape)
    c1=0
    c2=0


    s1 = np.random.randint(0, high=100000000) # 450
    s2 = np.random.randint(0, high=100000000) # 120
    s1 = 35977264
    s2 = 80473687
    k1=2
    k2=k1/4
    gen1 = np.random.RandomState(s1)
    gen2 = np.random.RandomState(s2)

    for t in range(duration):
        angle_update = 24 # 6deg per timestep

        c1 += angle_update
        c2 += angle_update

        c1_sample = c1 + np.degrees(gen1.vonmises(0, k1))
        c2_sample = c2 + np.degrees(gen2.vonmises(0, k2))

        rm.update_state(c1_sample,
                        c2_sample,
                        sm=angle_update,
                        plasticity=True
        )

    no_conf_r1 = rm.w_r1_epg
    no_conf_r2 = rm.w_r2_epg

    """
    Large conflict (179 degrees)
    """
    rm = RingModel({"n_r":n_r})
    rm = initialise(rm, c1=0, c2=179)

    # 'zero' the adjacency matrices for learning
    rm.w_r1_epg = np.zeros((8,n_r)) + base
    rm.w_r2_epg = np.zeros((8,n_r)) + base

    c1=0
    c2=179
    for t in range(duration):
        angle_update = 24 # 24deg per timestep

        c1+=angle_update
        c2+=angle_update

        rm.update_state(c1,
                        c2,
                        sm=angle_update,
                        plasticity=True
        )

    large_conf_r1 = rm.w_r1_epg
    large_conf_r2 = rm.w_r2_epg


    """
    One reliable, one not: c2 position constant w.r.t. agent
    """
    rm = RingModel({"n_r":n_r})
    rm = initialise(rm, c1=0, c2=0)

    # 'zero' the adjacency matrices for learning
    rm.w_r1_epg = np.zeros((8,n_r)) + base
    rm.w_r2_epg = np.zeros((8,n_r)) + base

    c1=0
    c2=0
    for t in range(duration):
        angle_update = 24 # 6deg per timestep

        c1+=angle_update

        rm.update_state(c1,
                        c2,
                        sm=angle_update,
                        plasticity=True
        )

    oron_r1 = rm.w_r1_epg
    oron_r2 = rm.w_r2_epg

    """
    Different cue strengths
    """
    rm = RingModel({"n_r":n_r})
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
                        w1=0.8,
                        w2=0.2,
                        sm=angle_update,
                        plasticity=True
        )

    diff_r1 = rm.w_r1_epg
    diff_r2 = rm.w_r2_epg


    """
    Plotting
    """
    fig = plt.figure(figsize=(4,8))

    subplots = fig.subplots(nrows=4,ncols=2,sharex=True,sharey=True)

    print(subplots)


    large_conf_axs = subplots[0]
    oron_axs = subplots[1]
    diff_axs = subplots[2]
    no_conf_axs = subplots[3] # noise_axs

    # Plot data
    cmaps = ['Reds','Blues']
    vmax = 0.25
    wmap = no_conf_axs[0].pcolormesh(no_conf_r1, vmin=0, vmax=vmax, cmap=cmaps[0], antialiased=True)
    wmap.set_edgecolor('face')
    wmap = no_conf_axs[1].pcolormesh(no_conf_r2, vmin=0, vmax=vmax, cmap=cmaps[1], antialiased=True)
    wmap.set_edgecolor('face')
    wmap = large_conf_axs[0].pcolormesh(large_conf_r1, vmin=0, vmax=vmax, cmap=cmaps[0], antialiased=True)
    wmap.set_edgecolor('face')
    wmap = large_conf_axs[1].pcolormesh(large_conf_r2, vmin=0, vmax=vmax, cmap=cmaps[1], antialiased=True)
    wmap.set_edgecolor('face')
    wmap = oron_axs[0].pcolormesh(oron_r1, vmin=0, vmax=vmax, cmap=cmaps[0], antialiased=True)
    wmap.set_edgecolor('face')
    wmap = oron_axs[1].pcolormesh(oron_r2, vmin=0, vmax=vmax, cmap=cmaps[1], antialiased=True)
    wmap.set_edgecolor('face')
    wmap1 = diff_axs[0].pcolormesh(diff_r1, vmin=0, vmax=vmax, cmap=cmaps[0], antialiased=True)
    wmap1.set_edgecolor('face')
    wmap2 = diff_axs[1].pcolormesh(diff_r2, vmin=0, vmax=vmax, cmap=cmaps[1], antialiased=True)
    wmap2.set_edgecolor('face')

    axins1 = no_conf_axs[0].inset_axes([0,-0.45,1,0.05])
    axins2 = no_conf_axs[1].inset_axes([0,-0.45,1,0.05])

    fig.colorbar(wmap1,
                 cax=axins1,
                 orientation='horizontal',
                 pad=1
                 # ax=cbar_axs[1]
    )
    fig.colorbar(wmap2,
                 cax=axins2,
                 orientation='horizontal',
                 pad=1
                 # ax=cbar_axs[1]
    )

    subtitles = [
        "A) Cues separated",
        "B) One cue useful",
        "C) Effect of weight",
        "D) Effect of reliability"
        ]

    xidx = 0
    yidx = 0
    for axs in subplots:
        for plot in axs:
            plot.set_aspect("equal")

            plot.set_ylim([8,0])
            plot.set_xlim([0,n_r])
            if xidx == 0: # Only for left-hand plots
                plot.set_ylabel("E-PG index")
                plot.text(-0.2, -0.5,subtitles[yidx], ha='left')

            if yidx == 3:
                cue = "Cue 1" if xidx == 0 else "Cue 2"
                plot.set_xlabel("R index ({})".format(cue))
                if n_r==8:
                    plot.set_xticks([0, 2, 4, 6, 8])
                else:
                    plot.set_xticks(np.arange(0,n_r,1))
            xidx += 1

        xidx = 0
        yidx +=1

#    plt.subplots_adjust(wspace=0.08, hspace=0)
    plt.savefig("plots/snapshot_comparison.svg", dpi=300, bbox_inches="tight")
#    plt.show()

if __name__ == "__main__":
    snapshot_comparison()
