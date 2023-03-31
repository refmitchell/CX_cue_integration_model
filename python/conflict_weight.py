"""
conflict_weight.py

Simulates different weight/reliability combinations to demonstrate the
requirement for differences in cue weight (amplitude) using the current learning
rule. Noise is not enough to create a weighting effect in the snapshot.
"""

import json
import numpy as np
import dict_key_definitions as keydefs
from extended_ring_model import *
from dict_key_definitions import rmkeys, decodekeys, simkeys, statskeys, simdatakeys, cwparamkeys
from dict_key_definitions import conflict_weight_reskeys as reskeys
import simulation_utilities as simutils
from test_utilities import angular_difference, circ_scatter

exp_params = dict()
exp_params[cwparamkeys.vary_strength] = (4, 4, 0.2, 0.8) # (k1, k2, w1, w2)
exp_params[cwparamkeys.vary_reliability] = (1, 4, 0.5, 0.5)
exp_params[cwparamkeys.vary_both] = (1, 4, 0.2, 0.8)
exp_params[cwparamkeys.vary_both_inv] = (4, 1, 0.2, 0.8)
exp_params[cwparamkeys.all_equal_k2] = (2, 2, 0.5, 0.5)
exp_params[cwparamkeys.all_equal_k4] = (4, 4, 0.5, 0.5)

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

    for exp in exp_params.keys():
        gen1 = np.random.RandomState(s1)
        gen2 = np.random.RandomState(s2)
        sm_gen = np.random.RandomState(sm_seed)
        placement_gen = np.random.RandomState(placement_seed)
        desired_gen = np.random.RandomState(desired_seed)

        # Local simulation parameters
        n = 100

        k1 = exp_params[exp][0]
        k2 = exp_params[exp][1]
        w1 = exp_params[exp][2]
        w2 = exp_params[exp][3]

        params = simutils.default_param_dict.copy()

        params[simkeys.cue_one_noise_gen] = gen1
        params[simkeys.cue_two_noise_gen] = gen2
        params[simkeys.sm_noise_gen] = sm_gen
        params[simkeys.placement_gen] = placement_gen
        params[simkeys.desired_direction_gen] = desired_gen
        params[simkeys.sm_noise] = True
        params[simkeys.k1] = k1
        params[simkeys.k2] = k2
        params[simkeys.s1] = s1
        params[simkeys.s2] = s2
        params[simkeys.w1] = w1
        params[simkeys.w2] = w2
        params[simkeys.sm_seed] = sm_seed
        params[simkeys.placement_angle_seed] = placement_seed
        params[simkeys.desired_angle_seed] = desired_seed

        data = dict()
        stats = dict()
        snapshots = dict()

        agents = simutils.initialise(n)

        # Phase 1: initial rolls
        params[simkeys.n_rolls] = 2
        params[simkeys.flatten_on_first] = True # Clear any previous snapshots
        simdata = simutils.perform_runs(agents, params)
        data[reskeys.phase1] = simdata[simdatakeys.world] # Orientation data
        snapshots[reskeys.phase1] = simdata[simdatakeys.snapshots] # Snapshots
        sns = snapshots[reskeys.phase1][0]

        desired_directions = []
        for individual in simdata[simdatakeys.world]:
            desired_directions.append(individual[0][0][4])

        # Phase 2: pre-conflict roll
        params[simkeys.flatten_on_first] = False # Allow snapshots to persist
        params[simkeys.n_rolls] = 1
        params[simkeys.desired] = desired_directions
        simdata = simutils.perform_runs(agents, params)
        data[reskeys.phase2] = simdata[simdatakeys.world] # Orientation data
        snapshots[reskeys.phase2] = simdata[simdatakeys.snapshots] # Snapshots

        # Phase 3: conflict roll
        params[simkeys.c2_offsets] = [120 for _ in agents]
        simdata = simutils.perform_runs(agents, params)
        data[reskeys.phase3] = simdata[simdatakeys.world] # Orientation data
        snapshots[reskeys.phase3] = simdata[simdatakeys.snapshots] # Snapshots

        # Phase 4: controll roll (cues re-aligned)
        params[simkeys.c2_offsets] = [0 for _ in agents]
        simdata = simutils.perform_runs(agents, params)
        data[reskeys.phase4] = simdata[simdatakeys.world] # Orientation data
        snapshots[reskeys.phase4] = simdata[simdatakeys.snapshots] # Snapshots

        print("Computing stats for {}".format(exp))

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
        with open("data_out/conflict_weight_{}_meta.json".format(exp), "w") as f:
            del params[simkeys.cue_one_noise_gen]
            del params[simkeys.cue_two_noise_gen]
            del params[simkeys.sm_noise_gen]
            del params[simkeys.placement_gen]
            del params[simkeys.desired_direction_gen]
            params[simkeys.n] = n
            json.dump(params, f)

        # Simulation data (can be used for constructing tracks)
        with open("data_out/conflict_weight_{}_data.json".format(exp), "w") as f:
            json.dump(data, f)

        # Simulation summary statistics.
        with open("data_out/conflict_weight_{}_stats.json".format(exp), "w") as f:
            json.dump(stats, f)
