"""
timeseries.py

This script generates the timeseries plots which demonstrate the basic
network characteristics (namely bump retention in 'darkness' and
R dominance over E-PG neurons).

Both the sm_peturbation and double_bump options are considered legacy
features. They are not used in the final paper. Double bumps could appear
during development but did not appear with the final network we used; this
appears to be a matter of tuning. The sm_peturbation option ended up being
superfluous as the desired property (R dominance) was demonstrated by the
no_sm option.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from extended_ring_model import *
from dict_key_definitions import rmkeys, decodekeys
from test_utilities import *
import argparse

def main(args):
    # Ring model for the timeseries
    n_r = 8
    # rm = dynamic_parameterise()
    rm = RingModel({rmkeys.n_r:n_r,
                    rmkeys.verbose:True,
                    rmkeys.d_w1:0.5,
                    rmkeys.d_w2:0.5})
    rm.initialise()

    # Sim parameters
    duration = args.duration
    window = args.test_window
    start = duration/2 - window/2
    end = duration/2 + window/2
    exp_type = args.type

    # Timeseries
    r1 = np.zeros((n_r,duration))
    r2 = np.zeros((n_r,duration))
    epg = np.zeros((8,duration))
    d7 = np.zeros((8, duration))
    pen = np.zeros((16,duration))
    peg = np.zeros((16,duration))

    a1 = 0
    a2 = 0
    sm = 0
    w1 = 0.5
    w2 = 0.5

    movement_generator = np.random.RandomState(439875902)

    # [Legacy] - double bumps did not appear in the final network.
    if exp_type == "double_bump":
        a2 = 179

    first = True
    ctr=0
    for t in range(duration):
        w1 = 0.5
        w2 = 0.5
        change = movement_generator.normal(loc=0, scale=20)

        if t >= start and t <= end:
            if exp_type == "cue_absence":
                # Update cues and sm, remove cue input to E-PGs
                # Bump should be driven by sm
                w1 = 0
                w2 = 0
                a1 += change
                a2 += change
                sm = change
            elif exp_type == "sm_peturbation": # [Legacy]
                # Do not update cues, update SM
                # Bump may be peturbed but should not move
                w1 = 0.5
                w2 = 0.5
                a1 = 180
                a2 = 180
                sm = change
            elif exp_type == "double_bump": # [Legacy]
                # Standard cue and sm update, positions overridden above.
                a1 += change
                a2 += change
                sm = change
            elif exp_type == "no_sm":
                # Shift cues, allow time for network to update.
                if first:
                    a1 += 180
                    a2 += 180
                    first = False
                sm = change
        else:
            a1 += change
            a2 += change
            sm = change

        # Update the ring model state
        rm.update_state(a1, a2, sm, w1=w1, w2=w2)

        epgs = rm.epg_rates.reshape(8)
        pens = rm.pen_rates.reshape(16)
        pegs = rm.peg_rates.reshape(16)
        d7s = rm.d7_rates.reshape(8)
        e = 0
        pn = 0
        pg = 0
        d = 0
        for idx in range(len(epgs)):
            if epgs[idx] > epgs[e]:
                e = idx
            if pens[idx] > pens[pn]:
                pn = idx
            if pegs[idx] > pegs[pg]:
                pg = idx
            if d7s[idx] > d7s[d]:
                d = idx



        es = []
        pns = []
        pgs = []
        ds = []
        for idx in range(len(epgs)):
            if epgs[idx] == epgs[e]:
                es.append(idx)
            if d7s[idx] == d7s[d]:
                ds.append(idx)


        for idx in range(len(epgs)*2):
            if pens[idx] == pens[pn]:
                pns.append(idx)
            if pegs[idx] == pegs[pg]:
                pgs.append(idx)

        epgs = np.zeros(len(epgs))
        pens = np.zeros(len(epgs)*2)
        pegs = np.zeros(len(epgs)*2)
        d7s = np.zeros(len(epgs))

        epgs[es] = 1
        pegs[pgs] = 1
        pegs[pgs] = 1
        pens[pns] = 1
        pens[pns] = 1
        d7s[ds] = 1

        # Logging
        r1[:,t] = rm.r1_rates.reshape(n_r)
        r2[:,t] = rm.r2_rates.reshape(n_r)
        epg[:,t] =  rm.epg_rates.reshape(8)
        d7[:,t] = rm.d7_rates.reshape(8)
        pen[:,t] = rm.pen_rates.reshape(16)
        peg[:,t] = rm.peg_rates.reshape(16)

        if exp_type == "double_bump":
            r1[:,t] = rm.r1_rates.reshape(n_r)
            r2[:,t] = rm.r2_rates.reshape(n_r)
            epg[:,t] = epgs
            d7[:,t] = d7s
            pen[:,t] = pens
            peg[:,t] = pegs


    mosaic = [["r1"],
              ["r2"],
              ["epg"],
              ["d7"],
              ["pen"],
              ["peg"]]

    fig,axs = plt.subplot_mosaic(mosaic, sharex=True)
    fig.set_size_inches((8,5))
    cmap='hot'

    wmap = axs["r1"].pcolormesh(r1, vmin=0, vmax=1, rasterized=True, cmap=cmap)
    wmap.set_edgecolor('face')
    axs["r1"].set_ylim([0,8])
    axs["r1"].set_yticks([0,8])
    axs["r1"].set_ylabel("R1")

    wmap = axs["r2"].pcolormesh(r2, vmin=0, vmax=1, rasterized=True, cmap=cmap)
    wmap.set_edgecolor('face')
    axs["r2"].set_ylim([0,8])
    axs["r2"].set_yticks([0,8])
    axs["r2"].set_ylabel("R2")

    wmap = axs["epg"].pcolormesh(epg, vmin=0, vmax=1, rasterized=True, cmap=cmap)
    wmap.set_edgecolor('face')
    axs["epg"].set_ylim([0,8])
    axs["epg"].set_yticks([0,8])
    axs["epg"].set_ylabel("E-PG")

    wmap = axs["d7"].pcolormesh(d7, vmin=0, vmax=1, rasterized=True, cmap=cmap)
    wmap.set_edgecolor('face')
    axs["d7"].set_ylim([0,8])
    axs["d7"].set_yticks([0,8])
    axs["d7"].set_ylabel(r"$\Delta$7")

    wmap = axs["pen"].pcolormesh(pen, vmin=0, vmax=1, rasterized=True, cmap=cmap)
    wmap.set_edgecolor('face')
    axs["pen"].set_ylim([0,16])
    axs["pen"].set_yticks([0,16])
    axs["pen"].set_ylabel("P-EN")

    wmap = axs["peg"].pcolormesh(peg, vmin=0, vmax=1, rasterized=True, cmap=cmap)
    wmap.set_edgecolor('face')
    axs["peg"].set_ylim([0,16])
    axs["peg"].set_yticks([0,16])
    axs["peg"].set_ylabel("P-EG")

    title = ""
    if exp_type == "cue_absence":
        title = "E-PG bump continues without external cues"
    elif exp_type == "no_sm":
        title = "E-PG bump primarily driven by cue input"

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig("plots/{}.pdf".format(exp_type), bbox_inches="tight")


def check_args(args):
    valid_types = ["double_bump",
                   "cue_absence",
                   "sm_peturbation",
                   "no_sm"]
    if args.type not in valid_types:
        valid_string = ""
        for t in valid_types: valid_string += "{} ".format(t)
        print("Error: valid types are: " + valid_string)
        sys.exit(-1)

    if args.duration <= 0:
        print("Error: strictly positive integer duration required.")
        sys.exit(-1)

    if args.test_window >= args.duration:
        print("Error: test window greater than overall duration.")
        sys.exit(-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Produce specified timeseries plots')
    parser.add_argument("-t",
                        "--type",
                        required=True,
                        type=str,
                        help="The type of experimental procedure to be performed")
    parser.add_argument("-d",
                        "--duration",
                        type=int,
                        default=1000,
                        help="The duration of the trial.")
    parser.add_argument("-w",
                        "--test_window",
                        required=False,
                        type=int,
                        default=200,
                        help="The duration of the test window (from the middle of the duration)."
    )

    args = parser.parse_args()
    check_args(args)
    main(args)
