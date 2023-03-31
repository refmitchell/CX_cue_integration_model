"""
precision.py

This routine generates the simulated precision data which is designed
to mimic our behavioural experiment (testing the precision of two
cues against one).
"""


import json
import numpy as np
import dict_key_definitions as keydefs
from extended_ring_model import *
from dict_key_definitions import rmkeys, decodekeys, simkeys, statskeys, simdatakeys
from dict_key_definitions import precision_reskeys as reskeys
import simulation_utilities as simutils
from test_utilities import angular_difference, circ_scatter

if __name__ == "__main__":
    # Init random generators
    s1 = np.random.randint(0, high=100000000)
    s2 = np.random.randint(0, high=100000000)

    s1 = 33672766
    s2 = 94314941

    print(s1)
    print(s2)

    sm_seed = 5437958
    placement_seed = 98437535
    desired_seed = 345702

    gen1 = np.random.RandomState(s1)
    gen2 = np.random.RandomState(s2)
    sm_gen = np.random.RandomState(sm_seed)
    placement_gen = np.random.RandomState(placement_seed)
    desired_gen = np.random.RandomState(desired_seed)

    # Local simulation parameters
    n = 20 # Number of beetles
    k1 = 4
    k2 = k1/4# 0.5
    wr1 = k1/(k1+k2) if (k1 + k2) >= 0 else 0.5 # Reliablity-based weights
    wr2 = 1 - wr1
    print(wr1)
    print(wr2)
    ws1 = 0.2 # Strength weights
    ws2 = 0.8

    # Initialise agents
    agents = simutils.initialise(n)

    # Parameter copy - set global parameters here (to be carried through each
    # simulation)
    params = simutils.default_param_dict.copy()
    params[simkeys.cue_one_noise_gen] = gen1
    params[simkeys.cue_two_noise_gen] = gen2
    params[simkeys.sm_noise_gen] = sm_gen
    params[simkeys.placement_gen] = placement_gen
    params[simkeys.desired_direction_gen] = desired_gen
    params[simkeys.n_rolls] = 10
    params[simkeys.sm_noise] = True
    params[simkeys.k1] = k1
    params[simkeys.k2] = k2
    params[simkeys.s1] = s1
    params[simkeys.s2] = s2
    params[simkeys.sm_seed] = sm_seed
    params[simkeys.placement_angle_seed] = placement_seed
    params[simkeys.desired_angle_seed] = desired_seed

    # Note: seeds are set but implicitly ignored as RandomStates are passed
    # (see simulation_utilities.py). They are set so that the parameter
    # dictionary contains the seeds used for each generator so that they
    # can be written out to a file.


    # Data and stats dictionary. One entry for each experiment, each of which
    # has an 'initial' and 'test' dataset. What these mean depends on the
    # experiment.
    data = dict()
    stats = dict()
    for key in reskeys.expkeys():
        data[key] = dict()
        stats[key] = dict()

    # Experiment 1: Single reliable cue plus unreliable cue (wbr)
    print("Running Experiment 1")
    agents = simutils.initialise(n)
    # 1st set
    params[simkeys.w1] = 1
    params[simkeys.w2] = 0
    params[simkeys.flatten_on_first] = True # Clear any previous snapshots
    data[reskeys.exp1][reskeys.initial] = simutils.perform_runs(agents, params)[simdatakeys.world]

    # 2nd set
    params[simkeys.w1] = wr1
    params[simkeys.w2] = wr2
    params[simkeys.flatten_on_first]=False # Continue from previous
    data[reskeys.exp1][reskeys.test] = simutils.perform_runs(agents, params)[simdatakeys.world]

    # Experiment 2: Single unreliable cue, add reliable cue (wbr)
    agents = simutils.initialise(n)
    print("Running Experiment 2")
    # 1st set
    params[simkeys.w1] = 0
    params[simkeys.w2] = 1
    params[simkeys.flatten_on_first] = True # Clear any previous snapshots
    data[reskeys.exp2][reskeys.initial] = simutils.perform_runs(agents, params)[simdatakeys.world]

    # 2nd set
    params[simkeys.w1] = wr1
    params[simkeys.w2] = wr2
    params[simkeys.flatten_on_first]=False # Continue from previous
    data[reskeys.exp2][reskeys.test] = simutils.perform_runs(agents, params)[simdatakeys.world]

    # Experiment 3: Two cues -> remove unreliable (wbr)
    agents = simutils.initialise(n)
    print("Running Experiment 3")
    # 1st set
    params[simkeys.w1] = wr1
    params[simkeys.w2] = wr2
    params[simkeys.flatten_on_first] = True # Clear any previous snapshots
    data[reskeys.exp3][reskeys.initial] = simutils.perform_runs(agents, params)[simdatakeys.world]

    # 2nd set
    params[simkeys.w1] = 1
    params[simkeys.w2] = 0
    params[simkeys.flatten_on_first]=False # Continue from previous
    data[reskeys.exp3][reskeys.test] = simutils.perform_runs(agents, params)[simdatakeys.world]

    # Experiment 4: Two cues -> remove reliable (wbr)
    agents = simutils.initialise(n)
    print("Running Experiment 4")
    # 1st set
    params[simkeys.w1] = wr1
    params[simkeys.w2] = wr2
    params[simkeys.flatten_on_first] = True # Clear any previous snapshots
    data[reskeys.exp4][reskeys.initial] = simutils.perform_runs(agents, params)[simdatakeys.world]

    # 2nd set
    params[simkeys.w1] = 0
    params[simkeys.w2] = 1
    params[simkeys.flatten_on_first]=False # Continue from previous
    data[reskeys.exp4][reskeys.test] = simutils.perform_runs(agents, params)[simdatakeys.world]

    # Experiment 5: Single reliable cue, add unreliable (wbs)
    agents = simutils.initialise(n)
    print("Running Experiment 5")
    # 1st set
    params[simkeys.w1] = 1
    params[simkeys.w2] = 0
    params[simkeys.flatten_on_first] = True # Clear any previous snapshots
    data[reskeys.exp5][reskeys.initial] = simutils.perform_runs(agents, params)[simdatakeys.world]

    # 2nd set
    params[simkeys.w1] = ws1
    params[simkeys.w2] = ws2
    params[simkeys.flatten_on_first]=False # Continue from previous
    data[reskeys.exp5][reskeys.test] = simutils.perform_runs(agents, params)[simdatakeys.world]

    # Experiment 6: Single unreliable cue, add reliable (wbs)
    agents = simutils.initialise(n)
    print("Running Experiment 6")
    # 1st set
    params[simkeys.w1] = 0
    params[simkeys.w2] = 1
    params[simkeys.flatten_on_first] = True # Clear any previous snapshots
    simdata = simutils.perform_runs(agents, params)#[simdatakeys.world]
    data[reskeys.exp6][reskeys.initial] = simdata[simdatakeys.world]# simutils.perform_runs(agents, params)[simdatakeys.world]
    snapshots = simdata[simdatakeys.snapshots]
    snapshots = snapshots[0]
    # print(len(snapshots))
    # for elem in snapshots:
    #     plt.subplot(121)
    #     plt.pcolormesh(elem[0], vmax=1)
    #     plt.subplot(122)
    #     plt.pcolormesh(elem[1], vmax=1)
    #     plt.show()

    # 2nd set
    params[simkeys.w1] = ws1
    params[simkeys.w2] = ws2
    params[simkeys.flatten_on_first]=False # Continue from previous
    simdata = simutils.perform_runs(agents, params)
    data[reskeys.exp6][reskeys.test] = simdata[simdatakeys.world]#simutils.perform_runs(agents, params)[simdatakeys.world]

    snapshots = simdata[simdatakeys.snapshots]
    snapshots = snapshots[0]
    # print(len(snapshots))
    # for elem in snapshots:
    #     plt.subplot(121)
    #     plt.pcolormesh(elem[0], vmax=1)
    #     plt.subplot(122)
    #     plt.pcolormesh(elem[1], vmax=1)
    #     plt.show()

    # Experiment 7: Two cues -> remove unreliable (wbs)
    agents = simutils.initialise(n)
    print("Running Experiment 7")
    # 1st set
    params[simkeys.w1] = ws1
    params[simkeys.w2] = ws2
    params[simkeys.flatten_on_first] = True # Clear any previous snapshots
    data[reskeys.exp7][reskeys.initial] = simutils.perform_runs(agents, params)[simdatakeys.world]

    # 2nd set
    params[simkeys.w1] = 1
    params[simkeys.w2] = 0
    params[simkeys.flatten_on_first]=False # Continue from previous
    data[reskeys.exp7][reskeys.test] = simutils.perform_runs(agents, params)[simdatakeys.world]

    # Experiment 8 Two cues -> remove reliable (wbs)
    agents = simutils.initialise(n)
    print("Running Experiment 8")
    # 1st set
    params[simkeys.w1] = ws1
    params[simkeys.w2] = ws2
    params[simkeys.flatten_on_first] = True # Clear any previous snapshots
    data[reskeys.exp8][reskeys.initial] = simutils.perform_runs(agents, params)[simdatakeys.world]

    # 2nd set
    params[simkeys.w1] = 0
    params[simkeys.w2] = 1
    params[simkeys.flatten_on_first]=False # Continue from previous
    data[reskeys.exp8][reskeys.test] = simutils.perform_runs(agents, params)[simdatakeys.world]

    print("All experiments complete, computing statistics")
    for key in data.keys():
        for run in reskeys.runkeys():
            stats[key][run] = simutils.summary_statistics(
                data[key][run],
                arena_size=params[simkeys.arena_size],
                step_size=params[simkeys.step_size]
            )

    # Write results
    # Simulation metadata (parameters). Note that w1, w2, and flatten_on_first
    # will all be meaningless here. To have this data be recoverable, one could
    # use the same dictionary-of-dictionaries construction used for data and
    # stats. This seemed like overkill. Generators are removed as they are not
    # serializable and the important parameter is the seed used.
    with open("data_out/precision_meta.json", "w") as f:
        del params[simkeys.cue_one_noise_gen]
        del params[simkeys.cue_two_noise_gen]
        del params[simkeys.sm_noise_gen]
        del params[simkeys.placement_gen]
        del params[simkeys.desired_direction_gen]
        json.dump(params, f)

    # Simulation data (can be used for constructing tracks)
    with open("data_out/precision_data.json", "w") as f:
        json.dump(data, f)

    # Simulation summary statistics.
    with open("data_out/precision_stats.json", "w") as f:
        json.dump(stats, f)

