"""
random_walk.py

Tests random walk learning with varied walk distribution (increasing
the angular velocity experienced) or varied learning time.
"""

import numpy as np
import matplotlib.pyplot as plt
from extended_ring_model import *
from dict_key_definitions import rmkeys, decodekeys

def generate_timeseries(duration=500,
                        learning_duration=200,
                        loc=0,
                        scale=20,
                        seed=984359074):
    n_r = 8
    rm = RingModel({rmkeys.n_r:n_r})
    rm.initialise()

    movement_gen = np.random.RandomState(seed)

    base = 1 / n_r

    start = duration/2 - learning_duration/2
    end = duration/2 + learning_duration/2
    mid = duration/2

    n_s = 3
    epgs = np.zeros((8,duration))
    snapshots = np.zeros((n_s,n_r,n_r))

    a1 = 0
    a2 = 0
    snap_ctr = 0

    for t in range(duration):
        sm = movement_gen.normal(scale=scale, loc=loc)
        a1 += sm
        a2 += sm
        plasticity=False

        if t >= start and t <= end:
            if (t == start or t == end) or t == mid:
                if t == start:
                    rm.w_r1_epg[:,:] = base
                    rm.w_r2_epg[:,:] = base
                snapshots[snap_ctr,:,:] = rm.w_r1_epg
                snap_ctr += 1

            plasticity=True

        rm.update_state(a1, a2, sm, plasticity=plasticity)
        epgs[:,t] = rm.epg_rates.reshape((8,))

    return {"epgs":epgs, "snapshots":snapshots}

def vary_time(seed=984359074):
    results = dict()
    learning_durations = [24, 50, 100]

    for ld in learning_durations:
        k = str(ld)
        results[k] = generate_timeseries(duration=200,
                                         learning_duration=ld,
                                         seed=seed)

    return results

def vary_noise(seed=984359074):
    results = dict()
    scales = [10, 20, 40]

    for s in scales:
        k = str(s)
        results[k] = generate_timeseries(duration=200,
                                         learning_duration=50,
                                         scale=s,
                                         seed=seed
        )
    return results


def make_plot(results, title="", filename="fig.pdf", sub_format="{}) {}", time=True):
    mosaic = [
        ['t1', 't1', 't1'],
        ['s11', 's12', 's13'],
        ['s11', 's12', 's13'],
        ['t2', 't2', 't2'],
        ['s21', 's22', 's23'],
        ['s21', 's22', 's23'],
        ['t3', 't3', 't3'],
        ['s31', 's32', 's33'],
        ['s31', 's32', 's33']
    ]

    fig, axs = plt.subplot_mosaic(mosaic, figsize=(8,12), sharey=True)

    snapshot_cmap = 'Greys_r'
    snapshot_vmin = 0
    snapshot_vmax = 0.25

    snapshot_titles = ['Start', 'Mid-learning', 'End']
    sub_titles = ['a', 'b', 'c']

    timeseries_cmap = 'hot'
    timeseries_vmin = 0
    timeseries_vmax = 1
    sub = 0

    for key in results.keys():
        epgs = results[key]["epgs"]
        snapshots = results[key]["snapshots"]
        sub_idx = str(sub+1) # subplot index
        timeseries_key = "t" + sub_idx
        snapshot_prefix = "s" + sub_idx
        axs[timeseries_key].set_title(sub_format.format(sub_titles[sub],
                                                        key))
        wmap = axs[timeseries_key].pcolormesh(epgs,
                                              vmin=timeseries_vmin,
                                              vmax=timeseries_vmax,
                                              cmap=timeseries_cmap)
        wmap.set_edgecolor('face')

        start_x = int(epgs.shape[1]/2 - 25)
        end_x = int(epgs.shape[1]/2 + 25)
        xs = range(start_x, end_x + 1)

        if time:
            start_x = int(epgs.shape[1]/2 - int(key)/2)
            end_x = int(epgs.shape[1]/2 + int(key)/2)
            xs = range(start_x, end_x + 1)

        axs[timeseries_key].fill_between(xs, 0, y2=9, color='red', alpha=0.5)
        axs[timeseries_key].set_ylim([0,8])

        axs[timeseries_key].set_ylabel("E-PG index")
        axs[timeseries_key].set_xlabel("Time")

        for ss in range(3):
            ss_idx = str(ss + 1)
            ss_key = snapshot_prefix + ss_idx

            axs[ss_key].set_title(snapshot_titles[ss])
            wmap = axs[ss_key].pcolormesh(snapshots[ss],
                                   vmin=snapshot_vmin,
                                   vmax=snapshot_vmax,
                                   cmap=snapshot_cmap)
            wmap.set_edgecolor('face')

            if ss == 0:
                axs[ss_key].set_ylim([8,0])
                axs[ss_key].set_yticks([0,2,4,6,8])
                axs[ss_key].set_ylabel("R index")
            if ss == 1:
                axs[ss_key].set_xlabel("E-PG index")
        sub += 1


    fig.tight_layout()
    if title != "":
        fig.suptitle(title)
    plt.savefig("plots/{}".format(filename), bbox_inches='tight')


if __name__ == "__main__":
    noise = vary_noise(seed=459820958)
    time = vary_time(seed=987564303)
    make_plot(noise,
              filename="rw_noise.pdf",
              sub_format="{}) Scale parameter = {}", time=False)
    make_plot(time,
              filename="rw_time.pdf",
              sub_format="{}) {} timesteps")
#    plt.show()








