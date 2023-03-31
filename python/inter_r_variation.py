"""
inter_r_variation.py

Simulates networks with different R group sizes to investigate
the effect of the unabalanced group size on cue conflict
outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from extended_ring_model import *
from dict_key_definitions import rmkeys, decodekeys
from test_utilities import *

def initialise(rm, c1=0, c2=0, w1=0.5, w2=0.5):
    """
    Initialisation routine.
    :param rm: a RingModel to initialise before learning
    :param c1: The start position of cue one
    :param c2: The start position of cue two
    :param w1: The weight given to cue one
    :param w2: The weight given to cue two
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

def learning_and_conflict(n_r1, n_r2, n, weights, conflicts):
    """
    Perform a dance and then generate the conflict data.

    :param n_r1: The number of neurons in R group one
    :param n_r2: The number of neurons in R group two
    :param n: The conflict resolution
    :param weights: The list of weights for cue one
    :param conflicts: The conflicts to be tested
    :return: The number of neurons in each group and the conflict data
    """
    duration = 15
    base = 1/n_r1 if n_r1 > n_r2 else 1/n_r2

    rm = RingModel({rmkeys.n_r1:n_r1,
                    rmkeys.n_r2:n_r2}
    )
    rm = initialise(rm, c1=0, c2=0)

    # 'zero' the adjacency matrices for learning
    rm.w_r1_epg = np.zeros((8,n_r1)) + base
    rm.w_r2_epg = np.zeros((8,n_r2)) + base

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

    n_weights = [rm.w_r1_epg, rm.w_r2_epg]
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

def comparison():
    """
    Create 4x2 figure showing conflict behaviour and snapshots for
    different numbers of R neurons.
    """
    n = 180
    conflicts = np.linspace(0,180,n)
    weights = np.arange(0.1, 1, 0.1)
    duration = 15 # Learning duration

    four_eight = learning_and_conflict(4, 8, n, weights, conflicts)
    eight_twelve = learning_and_conflict(8, 12, n, weights, conflicts)

    data_full = [four_eight, eight_twelve]

    """
    Plotting
    """
    fig = plt.figure(figsize=(7,3))
    subplots = fig.subplots(nrows=1,ncols=2)

    # Conflict data
    inc = 60
    ticks = np.arange(0,190+inc,inc)
    for idx in range(len(subplots)):
        conf_axs = subplots[idx]
        conf_data = data_full[idx][1]
        for jdx in range(len(conf_data)):
            condition = conf_data[jdx]
            w1 = weights[jdx]
            label = "{:.1}".format(w1)
            if jdx == 0:
                label = "w1 = {:.1}".format(w1)
            conf_axs.plot(condition, label=label)

        ticklabels = ["{}$^\degree$".format(x) for x in ticks]
        conf_axs.set_xticks(ticks, labels=ticklabels)
        conf_axs.set_yticks(ticks, labels=ticklabels)
        conf_axs.set_xlim([0,180])
        conf_axs.set_ylim([0,180])
        conf_axs.set_ylabel("Integrated angle")
        conf_axs.set_aspect("equal")
        conf_axs.set_xlabel("Cue conflict")
        if idx == 0:
            conf_axs.legend(loc="center",
                            ncols=len(weights),
                            bbox_to_anchor=(0.5, -0.03),
                            bbox_transform=fig.transFigure,
                            prop={"size":8}
            )
            conf_axs.set_title("$n_{R1} = 4, n_{R2} = 8$")
        else:
            conf_axs.set_title("$n_{R1} = 8, n_{R2} = 12$")
    fig.subplots_adjust(wspace=0.9, hspace=None)

    fig.savefig("plots/inter_r_variation.pdf", bbox_inches="tight")
#    plt.show()

if __name__ == "__main__":
    comparison()
