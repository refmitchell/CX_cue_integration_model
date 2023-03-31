"""
conflict_comparison.py

Generate the conflict comparison figure which compares the
ring model to the vector sum given by Murray and Morgenstern
(2010).

References:
Murray and Morgenstern(2010) - Cue combination on the circle and the sphere
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from dict_key_definitions import rmkeys, decodekeys
from extended_ring_model import *
from test_utilities import *

def conflict_comparison_figure(animation=False, show=False):
    """
    Generate the main conflict comparison figure.
    :param animation: Produce frames for an animation of the
                      learning process.
    """

    """
    Conflict comparison - 1x4 figure which shows the
    response of each model to increasing conflict
    """
    n_r = 8
    n_r1 = n_r
    n_r2 = n_r

    base = 1/n_r1 if n_r1 > n_r2 else 1/n_r2

    print("Neuron ratio: {}".format(n_r2/n_r1))
    print("Inv. ratio: {}".format(n_r1/n_r2))

    duration = 15 # Simulation duration

    # Plasticity animation frames
    r1_frames = np.zeros((duration,8,n_r1))
    r2_frames = np.zeros((duration,8,n_r2))

    # Starting angles
    c1=0
    c2=0
    w1=0.5
    w2=0.5

    rm = RingModel({rmkeys.n_r1:n_r1,
                    rmkeys.n_r2:n_r1})

    prm = RingModel({rmkeys.n_r1:n_r1,
                     rmkeys.n_r2:n_r1})

    # Additional model for no learning rotation
    rm_nosm = RingModel({rmkeys.n_r1:n_r1,
                         rmkeys.n_r2:n_r1})

    rm.initialise()
    prm.initialise()
    rm_nosm.initialise()

    # 'zero' the adjacency matrices prior to learning
    prm.w_r1_epg = np.zeros((8,n_r1)) + base
    prm.w_r2_epg = np.zeros((8,n_r2)) + base

    rm_nosm.w_r1_epg = np.zeros((8,n_r1)) + base
    rm_nosm.w_r2_epg = np.zeros((8,n_r2)) + base

    for t in range(duration):
        angle_update = 24 # 24deg per timestep

        c1+=angle_update
        c2+=angle_update
        prm.update_state(c1,
                         c2,
                         w1=w1,
                         w2=w2,
                         sm=angle_update,
                         plasticity=True
        )

        # Case with no movement during learning
        rm_nosm.update_state(0,
                             0,
                             w1=w1,
                             w2=w2,
                             sm=0,
                             plasticity=True
        )

        r1_frames[t] = prm.w_r1_epg
        r2_frames[t] = prm.w_r2_epg

    # Debugging
    weight_r1 = np.sum(prm.w_r1_epg)
    weight_r2 = np.sum(prm.w_r2_epg)
    weight_ratio = weight_r2/weight_r1

    print("Weight ratio: {}".format(weight_r2/weight_r1))

    #
    # Test all three models on conflicts
    #
    duration = 60
    cue_one = 0
    n = 180

    conflicts = np.linspace(0,180,n)
    weights = np.arange(0.1, 1, 0.1)

    # Store complete conflict data for each model
    ring_out_full = []
    prm_out_full = []
    mmcs_out_full = []
    nosm_out_full = []

    prm_mag_full = []

    # Cycle through each weight and conflict combination to generate
    # conflict plot data.
    for w1 in weights:
        rm.reset_rates()
        prm.reset_rates()
        rm_nosm.reset_rates()

        w2 = 1 - w1
        ring_out = []
        prm_out = []
        nosm_out = []
        mmcs_out = []
        prm_mag = []

        for cue_two in conflicts:
            # Ring model - expects degrees
            rm.update_state(cue_one,
                            cue_two,
                            sm=0,
                            w1=w1,
                            w2=w2,
                            plasticity=False)

            # Plastic ring model - expects degrees
            prm.update_state(cue_one,
                             cue_two,
                             sm=0,
                             w1=w1,
                             w2=w2,
                             plasticity=False)

            # Plastic ring model w/o learning rotation - expects degrees
            rm_nosm.update_state(cue_one,
                                 cue_two,
                                 sm=0,
                                 w1=w1,
                                 w2=w2,
                                 plasticity=False)

            static_t, static_r = rm.decode()[decodekeys.epg]
            plastic_t, plastic_r = prm.decode()[decodekeys.epg]
            nosm_t, nosm_r = rm_nosm.decode()[decodekeys.epg]

            #print("W1: {}, Mag: {}".format(w1, plastic_r))

            ring_out.append(np.rad2deg(static_t))
            prm_out.append(np.rad2deg(plastic_t) % 360)
            nosm_out.append(np.rad2deg(nosm_t) % 360)
            prm_mag.append(static_r)

            #
            # MMCS - expects radians
            #
            theta, r = mmcs(cue_one, np.deg2rad(cue_two), w1, w2)
            mmcs_out.append(np.rad2deg(theta))

        # Offset correction for plotting
        ring_offset = ring_out[0]
        ring_out = [x - ring_offset for x in ring_out]
        ring_out = [x % 360 for x in ring_out]

        prm_offset = prm_out[0]
        prm_out = [x - prm_offset for x in prm_out]
        prm_out = [x % 360 for x in prm_out]

        nosm_offset = nosm_out[0]
        nosm_out = [x - nosm_offset for x in nosm_out]
        nosm_out = [x % 360 for x in nosm_out]

        ring_out_full.append(ring_out)
        prm_out_full.append(prm_out)
        mmcs_out_full.append(mmcs_out)
        nosm_out_full.append(nosm_out)
        prm_mag_full.append(prm_mag)

    mosaic = [
        ['blank1', 'blank2', 'srm_w1', 'srm_w2', 'nrm_w1', 'nrm_w2', 'prm_w1', 'prm_w2'],
        ['mmcs',  'mmcs',  'srm',    'srm',    'nrm',    'nrm',    'prm',    'prm']
    ]

    # For ease of reference
    conf_plots = ['mmcs', 'srm', 'nrm', 'prm']
    snap_plots = ['srm_w1', 'srm_w2', 'nrm_w1', 'nrm_w2', 'prm_w1', 'prm_w2']

    fig, axs = plt.subplot_mosaic(mosaic, figsize=(8,4))

    # Plot conflict data
    for trial in range(len(ring_out_full)):
        if trial == 0:
            label = "w1 = {:.1f}".format(weights[trial])
        else:
            label = "{:.1f}".format(weights[trial])
        axs["mmcs"].plot(conflicts, mmcs_out_full[trial], label=label)
        axs["srm"].plot(conflicts, ring_out_full[trial], label=label)
        axs["nrm"].plot(conflicts, nosm_out_full[trial], label=label)
        axs["prm"].plot(conflicts, prm_out_full[trial], label=label)

    # Plot snapshots
    cmap = 'Greys_r'
    vmax = 0.25
    vmin=0
    wmap = axs['srm_w1'].pcolormesh(rm.w_r1_epg, cmap=cmap, vmin=vmin,  vmax=vmax, rasterized=True)
    wmap.set_edgecolor('face')
    wmap = axs['srm_w2'].pcolormesh(rm.w_r2_epg, cmap=cmap, vmin=vmin,  vmax=vmax, rasterized=True)
    wmap.set_edgecolor('face')
    wmap = axs['nrm_w1'].pcolormesh(rm_nosm.w_r1_epg, cmap=cmap, vmin=vmin,  vmax=vmax, rasterized=True)
    wmap.set_edgecolor('face')
    wmap = axs['nrm_w2'].pcolormesh(rm_nosm.w_r2_epg, cmap=cmap, vmin=vmin,  vmax=vmax, rasterized=True)
    wmap.set_edgecolor('face')
    wmap = axs['prm_w1'].pcolormesh(prm.w_r1_epg, cmap=cmap, vmin=vmin,  vmax=vmax, rasterized=True)
    wmap.set_edgecolor('face')
    wmap = axs['prm_w2'].pcolormesh(prm.w_r2_epg, cmap=cmap, vmin=vmin,  vmax=vmax, rasterized=True)
    wmap.set_edgecolor('face')

    # Deal with blank axes
    axs['blank1'].axis('off')
    axs['blank2'].axis('off')

    # Universal formatting
    fig.tight_layout()
    plt.subplots_adjust(hspace=-0.3, wspace=0.4)
    for key in axs.keys():
        axs[key].set_aspect('equal')

    # Conflict formatting
    for key in conf_plots:
        if key == "mmcs":
            axs[key].legend(ncols=len(weights),
                            prop={'size':8},
                            loc=10,
                            bbox_to_anchor=(0.5, 0),
                            bbox_transform=fig.transFigure)
        axs[key].set_xlim([0,180])
        axs[key].set_ylim([0,180])
        axs[key].set_xticks([0,60,120,180],
                            labels=["$0^\degree$",
                                    "$60^\degree$",
                                    "$120^\degree$",
                                    "$180^\degree$"])
        axs[key].set_yticks([0,60,120,180],
                            labels=["$0^\degree$",
                                    "$60^\degree$",
                                    "$120^\degree$",
                                    "$180^\degree$"]
                            )

        axs[key].set_xlabel("Cue conflict")
        if key != 'mmcs':
            axs[key].set_yticks([])
        else:
            axs[key].set_ylabel("Integrated angle")

    y_offset=1.8
    axs['srm'].set_title("B) Default mapping",y=y_offset,va='top')
    axs['nrm'].set_title("C) Learning:\n without rotation",y=y_offset, va='top')
    axs['prm'].set_title("D) Learning:\n with rotation",y=y_offset, va='top')
    axs['mmcs'].set_title("A) Vector sum",y=y_offset, va='top')

    # Snapshot formatting
    for key in snap_plots:
        axs[key].set_xlim([0,8])
        axs[key].set_ylim([8,0])
        axs[key].set_xticks([])
        axs[key].set_yticks([])

    #plt.savefig("plots/conflict_comparison.pdf", bbox_inches="tight")
    plt.savefig("plots/conflict_comparison.png", bbox_inches="tight", dpi=400)

    if animation:
        #
        # Animation plotting
        #
        fig = plt.figure()
        w1_ax = plt.subplot(121)
        w2_ax = plt.subplot(122)
        fig.tight_layout()

        for t in range(len(r1_frames)):
            r1_epg = r1_frames[t]
            r2_epg = r2_frames[t]
            filename = "{:05d}.png".format(t)

            w1_ax.clear()
            w2_ax.clear()

            # Data
            w1_ax.pcolormesh(r1_epg, vmin=0, vmax=1, cmap='hot')
            w2_ax.pcolormesh(r2_epg, vmin=0, vmax=1, cmap='hot')

            # Appearance
            # W1
            w1_ax.set_ylabel("E-PG index")
            w1_ax.set_xlabel("R-index")
            w1_ax.set_ylim([8,0])
            w1_ax.set_xlim([0,n_r1])
            w1_ax.set_title("R1 -> E-PG adjacency")

            # W2
            w2_ax.set_ylabel("E-PG index")
            w2_ax.set_xlabel("R-index")
            w2_ax.set_ylim([8,0])
            w2_ax.set_xlim([0,n_r2])
            w2_ax.set_title("R2 -> E-PG adjacency")

            plt.savefig("plots/class_weight_frames/{}".format(filename))

            print("{}, done.".format(filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--animation",
                        required=False,
                        default=False,
                        action='store_true')
    parser.add_argument("--show",
                        required=False,
                        default=False,
                        action='store_true')
    args = parser.parse_args()

#    mmcs_vs_static_ring()
#    plasticity_tuning(animation=False)
#    simple_timeseries()
    conflict_comparison_figure(animation=args.animation, show=args.show)
#    rotation_comparison_figure(animation=True)




