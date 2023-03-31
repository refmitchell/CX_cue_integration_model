"""
modality_transfer.py

Mimic the modality transfer experiment from Dacke et al. (2019).

References:
Dacke et al. (2019) - Multimodal cue integration in the dung beetle compass
- Transfer of information between different compasses
"""

import json
import numpy as np
import dict_key_definitions as keydefs
from extended_ring_model import *
from dict_key_definitions import rmkeys, decodekeys, simkeys, statskeys, simdatakeys
from dict_key_definitions import modality_transfer_reskeys as reskeys
import simulation_utilities as simutils
from test_utilities import angular_difference, circ_scatter

if __name__ == "__main__":
    # Init random generators
    s1 = np.random.randint(0, high=100000000)
    s2 = np.random.randint(0, high=100000000)

    s1 = 73866205
    s2 = 87415498

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
    n = 40 # number of beetles (Dacke et al., 2019)
    k1 = 2 # Both cues reliable
    k2 = 2

    # Parameter copy - set global parameters here (to be carried through each
    # simulation)
    params = simutils.default_param_dict.copy()
    params[simkeys.cue_one_noise_gen] = gen1
    params[simkeys.cue_two_noise_gen] = gen2
    params[simkeys.sm_noise_gen] = sm_gen
    params[simkeys.placement_gen] = placement_gen
    params[simkeys.desired_direction_gen] = desired_gen
    params[simkeys.n_rolls] = 1 # Rolls are performed individually for ease of analysis
    params[simkeys.sm_noise] = True
    params[simkeys.k1] = k1
    params[simkeys.k2] = k2
    params[simkeys.s1] = s1
    params[simkeys.s2] = s2
    params[simkeys.sm_seed] = sm_seed
    params[simkeys.placement_angle_seed] = placement_seed
    params[simkeys.desired_angle_seed] = desired_seed

    data = dict()
    stats = dict()
    snapshots = dict()

    # Data dictionary for each roll
    # Have snapshots for each agent for each roll
    # Have tracks (and exits) for each agent for each roll

    # Agents initialised at the start, state and snapshot memory persist after
    # first roll. Any parameters not explicitly set before perform_runs is called
    # are carried over from the previous.
    agents = simutils.initialise(n)

    # Roll 1: cue 1 only.
    params[simkeys.w1] = 1
    params[simkeys.w2] = 0
    params[simkeys.flatten_on_first] = True # Clear any previous snapshots
    simdata = simutils.perform_runs(agents, params)
    data[reskeys.roll1] = simdata[simdatakeys.world] # Orientation data
    snapshots[reskeys.roll1] = simdata[simdatakeys.snapshots] # Snapshots

    # Extract desired directions from first roll to pass through
    # remaining experiments.
    desired_directions = []
    for individual in simdata[simdatakeys.world]:
        desired_directions.append(individual[0][0][4])

    # Roll 2: cues together
    params[simkeys.w1] = 0.5
    params[simkeys.w2] = 0.5
    params[simkeys.desired] = desired_directions
    # Dacke et al. (2019) present new cue at 90deg
    params[simkeys.c2_offsets] = [90 for _ in agents]
    params[simkeys.flatten_on_first] = False # Allow snapshots to persist
    simdata = simutils.perform_runs(agents, params)
    data[reskeys.roll2] = simdata[simdatakeys.world] # Orientation data
    snapshots[reskeys.roll2] = simdata[simdatakeys.snapshots] # Snapshots

    # Roll 3: cues together
    simdata = simutils.perform_runs(agents, params)
    data[reskeys.roll3] = simdata[simdatakeys.world] # Orientation data
    snapshots[reskeys.roll3] = simdata[simdatakeys.snapshots] # Snapshots

    # Roll 4: cue 2 only
    params[simkeys.w1] = 0
    params[simkeys.w2] = 1
    simdata = simutils.perform_runs(agents, params)
    data[reskeys.roll4] = simdata[simdatakeys.world] # Orientation data
    snapshots[reskeys.roll4] = simdata[simdatakeys.snapshots] # Snapshots

    print("All rolls complete, computing statistics")
    for key in data.keys():
        stats[key] = simutils.summary_statistics(
            data[key],
            arena_size=params[simkeys.arena_size],
            step_size=params[simkeys.step_size]
        )

    # Write results
    # Simulation metadata (parameters). Note that w1, w2, and flatten_on_first
    # will all be meaningless here. Generators are removed as they are not
    # serializable and the important parameter is the seed used.
    with open("data_out/modality_transfer_meta.json", "w") as f:
        del params[simkeys.cue_one_noise_gen]
        del params[simkeys.cue_two_noise_gen]
        del params[simkeys.sm_noise_gen]
        del params[simkeys.placement_gen]
        del params[simkeys.desired_direction_gen]
        json.dump(params, f)

    # Simulation data (can be used for constructing tracks)
    with open("data_out/modality_transfer_data.json", "w") as f:
        json.dump(data, f)

    # Simulation summary statistics.
    with open("data_out/modality_transfer_stats.json", "w") as f:
        json.dump(stats, f)
