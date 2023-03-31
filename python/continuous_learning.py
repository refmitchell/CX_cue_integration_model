"""
continuous_learning.py

This script provides a simulation routine which tests how the ring
model behaves where plasticity is enabled during behaviour.  This was
written and tested before the insight provided by Dan et al. (2022)
and Fisher et al. (2022); namely, that angular velocity appears to
regulate plasticity between R and E-PG neurons. We had no reason to
make any assumptions about the up- or down-regulating of plasticity
beyond those we made about the dance, so we tested constant plasticity.

Note that the different learning rates are only applied after the dance
happens (i.e. the dance always uses the default learning rate specified
in extended_ring_model.py).

References:
Dan et al. (2022) - Flexible control of behavioral variability mediated by an internal representation of head direction
Fisher et al. (2022) - Dopamine promotes head direction plasticity during orienting movements

"""

from extended_ring_model import *
from dict_key_definitions import rmkeys, decodekeys

import numpy as np
import matplotlib.pyplot as plt

def produce_data():
    # Init a list of ring models, then for each
    # Run in static mode for t_static
    # Run dance routine
    # Run in static mode for t_static
    # Run in plastic mode for t_plastic
    # Slowly introduce cue separation (up to c_max)
    # Continue running in plastic mode for t_plastic
    # Introduce conflict (immediate separation) and observe for t_conf.

    agents = []
    learning_rates = [0.1, 0.01, 0.001]
    # learning_rates = [0.0005]
    # learning_rates = [0.001]
    for r in learning_rates:
        agents.append(RingModel())

    heading_traces = []
    mapping_traces = []
    data = []

    r1_mappings = []
    r2_mappings = []

    # Basic init
    for a in agents:
        a.initialise()

    t_static=100 # Used twice
    t_dance=15
    t_plastic=1000 # Used twice
    t_plastic_2=5000
    t_drift=1000
    t_conflict=8000

    # Total duration
    t_global = 2*t_static + t_dance + t_plastic + t_plastic_2 + t_drift + t_conflict

    # C2 drift (degrees per timestep)
    drift = 180/t_drift

    # C2 conflict (degrees)
    c2_conflict = 179

    for a_idx in range(len(agents)):
        a = agents[a_idx]
        lr = learning_rates[a_idx]
        r1_map_trace = np.zeros((a.n_r1, t_global))
        r2_map_trace = np.zeros((a.n_r2, t_global))
        epg_trace = np.zeros((t_global,))
        t = 0 # global time index

        c1 = 0
        c2 = 0
        w1 = 0.5
        w2 = 0.5
        seed = 324898327
        movement_scale = 10
        movement_generator = np.random.RandomState(seed=seed)

        # First stage (random walk, static connections)
        for step in range(t_static):
            change = movement_generator.normal(loc=0, scale=movement_scale)
            c1 += change
            c2 += change
            a.update_state(c1, c2, sm=change, w1=w1, w2=w2, plasticity=False)

            # Log
            state = a.decode()
            r1_map = [x for (x,_) in state[decodekeys.r1_epg]]
            r2_map = [x for (x,_) in state[decodekeys.r2_epg]]
            epg_trace[t] = np.degrees(state[decodekeys.epg][0])
            r1_map_trace[:,t] = r1_map
            r2_map_trace[:,t] = r2_map
            t += 1

        # Dance
        dance_onset=t
        base = 1/a.n_r1 if a.n_r1 > a.n_r2 else 1/a.n_r2
        a.w_r1_epg = np.zeros((8,a.n_r1)) + base
        a.w_r2_epg = np.zeros((8,a.n_r2)) + base
        for step in range(t_dance):
            angle_update = 24 # 24deg per timestep

            c1+=angle_update
            c2+=angle_update

            a.update_state(c1,
                           c2,
                           w1=w1,
                           w2=w2,
                           sm=angle_update,
                           plasticity=True
            )

            # Log
            state = a.decode()
            r1_map = [x for (x,_) in state[decodekeys.r1_epg]]
            r2_map = [x for (x,_) in state[decodekeys.r2_epg]]
            epg_trace[t] = np.degrees(state[decodekeys.epg][0])
            r1_map_trace[:,t] = r1_map
            r2_map_trace[:,t] = r2_map

            t += 1

        static_two_onset=t
        # Second stage (random walk, static connections)
        for step in range(t_static):
            change = movement_generator.normal(loc=0, scale=movement_scale)
            c1 += change
            c2 += change
            a.update_state(c1, c2, sm=change, w1=w1, w2=w2, plasticity=False)

            state = a.decode()
            r1_map = [x for (x,_) in state[decodekeys.r1_epg]]
            r2_map = [x for (x,_) in state[decodekeys.r2_epg]]
            epg_trace[t] = np.degrees(state[decodekeys.epg][0])
            r1_map_trace[:,t] = r1_map
            r2_map_trace[:,t] = r2_map

            t += 1

        a.learning_rate = lr # Update learning rate
        cl_onset=t
        # Third stage (random walk, plasticity enabled, continuous learning)
        for step in range(t_plastic):
            change = movement_generator.normal(loc=0, scale=20)
            c1 += change
            c2 += change
            a.update_state(c1, c2, sm=change, w1=w1, w2=w2, plasticity=True)

            state = a.decode()
            r1_map = [x for (x,_) in state[decodekeys.r1_epg]]
            r2_map = [x for (x,_) in state[decodekeys.r2_epg]]
            epg_trace[t] = np.degrees(state[decodekeys.epg][0])
            r1_map_trace[:,t] = r1_map
            r2_map_trace[:,t] = r2_map

            t += 1

        drift_onset=t
        # Fourth stage (random walk, continuous learning, C2 drift)
        for step in range(t_drift):
            change = movement_generator.normal(loc=0, scale=movement_scale)
            c1 += change
            c2 += (change + drift)
            a.update_state(c1, c2, sm=change, w1=w1, w2=w2, plasticity=True)

            state = a.decode()
            r1_map = [x for (x,_) in state[decodekeys.r1_epg]]
            r2_map = [x for (x,_) in state[decodekeys.r2_epg]]
            epg_trace[t] = np.degrees(state[decodekeys.epg][0])
            r1_map_trace[:,t] = r1_map
            r2_map_trace[:,t] = r2_map

            t += 1

        # plt.subplot(121)
        # plt.pcolormesh(a.w_r1_epg)
        # plt.subplot(122)
        # plt.pcolormesh(a.w_r2_epg)
        # plt.show()
        
        cl_two_onset = t
        # Fifth stage (as in Stage 3)
        for step in range(t_plastic_2):
            change = movement_generator.normal(loc=0, scale=movement_scale)
            c1 += change
            c2 += change
            a.update_state(c1, c2, sm=change, w1=w1, w2=w2, plasticity=True)

            state = a.decode()
            r1_map = [x for (x,_) in state[decodekeys.r1_epg]]
            r2_map = [x for (x,_) in state[decodekeys.r2_epg]]
            epg_trace[t] = np.degrees(state[decodekeys.epg][0])
            r1_map_trace[:,t] = r1_map
            r2_map_trace[:,t] = r2_map

            t += 1

        # Sixth stage (conflict)
        c2 += c2_conflict
        conflict_onset = t
        for step in range(t_conflict):
            change = movement_generator.normal(loc=0, scale=movement_scale)
            c1 += change
            c2 += change
            a.update_state(c1, c2, sm=change, w1=w1, w2=w2, plasticity=True)

            state = a.decode()
            r1_map = [x for (x,_) in state[decodekeys.r1_epg]]
            r2_map = [x for (x,_) in state[decodekeys.r2_epg]]
            epg_trace[t] = np.degrees(state[decodekeys.epg][0])
            r1_map_trace[:,t] = r1_map
            r2_map_trace[:,t] = r2_map

            t += 1

        data.append([epg_trace, r1_map_trace, r2_map_trace])
        r1_mappings.append(a.w_r1_epg)
        r2_mappings.append(a.w_r2_epg)

    #
    # Plotting
    #
    fig, axs = plt.subplots(nrows=len(data)*3, ncols=1, sharex=True, figsize=(12, 4*len(data)))

    for idx in range(len(data)):
        heading_axs = axs[idx*3]
        r1_map_axs = axs[idx*3 + 1]
        r2_map_axs = axs[idx*3 + 2]

        heading_trace = data[idx][0]
        r1_map_trace = data[idx][1]
        r2_map_trace = data[idx][2]

        r1_map_trace = np.degrees(r1_map_trace)
        r2_map_trace = np.degrees(r2_map_trace)

        heading_trace = np.array(np.unwrap(heading_trace,
                                           discont=180,
                                           period=360))

        # Shift things for a slightly nicer plot
        shifts = np.linspace(-5*360, 360, 7)
        for shift in shifts:
            heading_axs.plot(heading_trace+shift, color='tab:grey', linewidth=0.5, zorder=0)

        heading_axs.fill_between(list(range(t_static, t_static + t_dance)),
                                 -180,
                                 y2 = 180,
                                 color='tab:blue',
                                 alpha=0.5
        )
        heading_axs.vlines([conflict_onset,
                            drift_onset,
                            cl_onset,
                            cl_two_onset],
                           -180, 180,
                           color=['green', 'magenta', 'red', 'red'],
                           alpha=[1, 1, 1, 1])

        heading_axs.set_ylim([-180,180])
        heading_axs.set_ylabel("Heading")

        wmap = r1_map_axs.pcolormesh(r1_map_trace,cmap='twilight', rasterized=True)
        cbar_axs = r1_map_axs.inset_axes([1.05, 0, 0.01, 1])
        fig.colorbar(wmap,
                     cax=cbar_axs,
                     orientation='vertical')
        r1_map_axs.vlines([conflict_onset,
                            drift_onset,
                            cl_onset,
                            cl_two_onset],
                           0, 8,
                           color=['green', 'magenta', 'red', 'red'],
                          alpha = [1, 1, 1, 1])
        r1_map_axs.set_ylabel("$\eta = {}$\n\n R1 map".format(learning_rates[idx]))

        wmap = r2_map_axs.pcolormesh(r2_map_trace,cmap='twilight', rasterized=True)
        cbar_axs = r2_map_axs.inset_axes([1.05, 0, 0.01, 1])
        fig.colorbar(wmap,
                     cax=cbar_axs,
                     orientation='vertical')
        r2_map_axs.vlines([conflict_onset,
                            drift_onset,
                            cl_onset,
                            cl_two_onset],
                           -0, 8,
                           color=['green', 'magenta', 'red', 'red'],
                          alpha = [1,1,1,1])
        r2_map_axs.set_ylabel("R2 map")

        if idx == len(data) - 1:
            r2_map_axs.set_xlabel("Time")


    # ss_fig, axs = plt.subplots(nrows=len(data), ncols=2, sharex=True, sharey=True)
    # for idx in range(len(r1_mappings)):
    #     axs[0].pcolormesh(r1_mappings[idx], vmax=0.25)
    #     axs[1].pcolormesh(r2_mappings[idx], vmax=0.25)
    plt.savefig("plots/continuous.pdf",bbox_inches="tight", dpi=300)
    plt.show()

    """
    data[x] = data for agent x
    data[x][0] = heading trace for agent x
    data[x][1] = mapping trace for agent x
    """
    return data

if __name__ == "__main__":
    produce_data()
