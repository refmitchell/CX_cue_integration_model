"""
conflict_shav.py

Simulates the cue conflict paradigm from Shaverdian et al. (2022).
The simulation routine can either use our strength model
(strength_model.py) to set cue weights, or the adjusted reliability
weight model from Shaverdian et al. (2022) (specify -s option).

Cue reliability is determined using the estimation functions
from Shaverdian et al. (2022).

References:
Shaverdian et al. (2022) - Weighted cue integration for straight-line
                           orientation
"""

import argparse
import json
import numpy as np
import dict_key_definitions as keydefs
from extended_ring_model import *
from dict_key_definitions import rmkeys, decodekeys, simkeys, statskeys, simdatakeys, dackeparams
from dict_key_definitions import conflict_shav_reskeys as reskeys
import simulation_utilities as simutils
from test_utilities import angular_difference, circ_scatter, kappa
import strength_model as sm


def generate_conditions():
    conflicts = [60,120]
    elevations = [45,60,75,86]
    wind_speeds = [1.25,2.5]

    conditions = []
    for e in elevations:
        for s in wind_speeds:
            if e == 45 and s == 1.25:
                continue # 45deg elevation and 1.25 m/s windspeed wasn't tested.
            conditions.append((e,s))

    return conditions, conflicts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s",
                        "--shaverdian2022",
                        action="store_true",
                        required=False,
                        help="Plot kappa estimation from Shaverdian et al. (2022)")
    args = parser.parse_args()
    shav=args.shaverdian2022

    if shav:
        print("info: using weight model from Shaverdian et al. (2022)")
    else:
        print("info: using strength-based weight model")
    conditions, conflicts = generate_conditions()

    # Init random generators
    s1 = np.random.randint(0, high=100000000)
    s2 = np.random.randint(0, high=100000000)

    # Good seeds for illustration
    s1 = 61374006
    s2 = 94904688

    print(s1)
    print(s2)

    sm_seed = 5437958
    placement_seed = 98437535
    desired_seed = 345702
    conflict_order_seed = 94374982

    gen1 = np.random.RandomState(s1)
    gen2 = np.random.RandomState(s2)
    sm_gen = np.random.RandomState(sm_seed)
    placement_gen = np.random.RandomState(placement_seed)
    desired_gen = np.random.RandomState(desired_seed)

    # Local generator to randomise conflict ordering
    conflict_order_gen = np.random.RandomState(conflict_order_seed)

    with open("data_out/weights_from_shav.json") as f:
        weight_dict = json.load(f)

    for exp in conditions:
        # Local simulation parameters
        n = 30

        elevation_degrees = exp[0]
        elevation_radians = np.radians(elevation_degrees)
        wind_speed = exp[1]

        k1, r1 = sm.solar_kappa_estimator(elevation_radians)
        k2, r2 = sm.wind_kappa_estimator(wind_speed)



        weight_dict_key = "{}-{:.2f}".format(elevation_degrees, wind_speed)

        # Compute the relative weight of the light using the vector
        # projection/sigmoid strength models.
        w1 = sm.relative_strength(elevation_radians, wind_speed, shav=shav)
        w2 = 1 - w1

        print("Exp: {}".format(weight_dict_key))
        print("K1 = {}".format(k1))
        print("K2 = {}".format(k2))
        print("R1 = {}".format(r1))
        print("R2 = {}".format(r2))
        print("W1 = {}".format(w1))
        print("W2 = {}".format(w2))

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

        # Generate conflicts, each agent experiences two in a pseudorandom order
        first_conflict_indices = conflict_order_gen.randint(2, size=n) # 1 or 0
        conflict_orders = [ # Can be stored with data for plotting
            [conflicts[i], conflicts[(i - 1)**2]] for i in first_conflict_indices
        ]

        # Lists of conflicts to be experienced by each agent
        initial_conflicts = [ x for [x,_] in conflict_orders]
        secondary_conflicts = [ x for [_,x] in conflict_orders]

        data[reskeys.conflict_orders] = conflict_orders

        # Phase 1: initial rolls
        params[simkeys.n_rolls] = 1
        params[simkeys.flatten_on_first] = True # Clear any previous snapshots
        simdata = simutils.perform_runs(agents, params)
        data[reskeys.phase1] = simdata[simdatakeys.world] # Orientation data
        snapshots[reskeys.phase1] = simdata[simdatakeys.snapshots] # Snapshots

        desired_directions = []
        for individual in simdata[simdatakeys.world]:
            desired_directions.append(individual[0][0][4])

        # Phase 2, zero conflict
        params[simkeys.n_rolls] = 1
        params[simkeys.desired] = desired_directions
        params[simkeys.flatten_on_first] = False
        simdata = simutils.perform_runs(agents, params)
        data[reskeys.phase2] = simdata[simdatakeys.world] # Orientation data
        snapshots[reskeys.phase2] = simdata[simdatakeys.snapshots] # Snapshots

        # Phase 3: pre-conflict roll
        params[simkeys.n_rolls] = 1
        simdata = simutils.perform_runs(agents, params)
        data[reskeys.phase3] = simdata[simdatakeys.world] # Orientation data
        snapshots[reskeys.phase3] = simdata[simdatakeys.snapshots] # Snapshots

        # Phase 4: initial conflict roll
        params[simkeys.c2_offsets] = initial_conflicts
        simdata = simutils.perform_runs(agents, params)
        data[reskeys.phase4] = simdata[simdatakeys.world] # Orientation data
        snapshots[reskeys.phase4] = simdata[simdatakeys.snapshots] # Snapshots

        # Phase 5: control roll (cues re-aligned)
        params[simkeys.c2_offsets] = [] # Back to original position
        simdata = simutils.perform_runs(agents, params)
        data[reskeys.phase5] = simdata[simdatakeys.world] # Orientation data
        snapshots[reskeys.phase5] = simdata[simdatakeys.snapshots] # Snapshots

        # Phase 6: second conflict roll
        params[simkeys.c2_offsets] = secondary_conflicts
        simdata = simutils.perform_runs(agents, params)
        data[reskeys.phase6] = simdata[simdatakeys.world] # Orientation data
        snapshots[reskeys.phase6] = simdata[simdatakeys.snapshots] # Snapshots

        # Phase 7: control
        params[simkeys.c2_offsets] = [] # Back to original position
        simdata = simutils.perform_runs(agents, params)
        data[reskeys.phase7] = simdata[simdatakeys.world] # Orientation data
        snapshots[reskeys.phase7] = simdata[simdatakeys.snapshots] # Snapshots

        # Phase 8: control
        simdata = simutils.perform_runs(agents, params)
        data[reskeys.phase8] = simdata[simdatakeys.world] # Orientation data
        snapshots[reskeys.phase8] = simdata[simdatakeys.snapshots] # Snapshots

        expstring = "{}-{}".format(elevation_degrees, wind_speed)
        print("Computing stats for {}".format(expstring))

        for key in data.keys():
            if key == reskeys.conflict_orders:
                continue

            stats[key] = simutils.summary_statistics(
                data[key],
                arena_size=params[simkeys.arena_size],
                step_size=params[simkeys.step_size]
            )

        # Write results
        # Simulation metadata (parameters). Note that w1, w2, and flatten_on_first
        # will all be meaningless here. Generators are removed as they are not
        # serializable and the important parameter is the seed used.
        wmspec = "swm" if shav else "pwm"
        with open("data_out/conflict_shav_{}_meta_{}.json".format(wmspec,expstring), "w") as f:
            del params[simkeys.cue_one_noise_gen]
            del params[simkeys.cue_two_noise_gen]
            del params[simkeys.sm_noise_gen]
            del params[simkeys.placement_gen]
            del params[simkeys.desired_direction_gen]
            params[simkeys.n] = n
            json.dump(params, f)

        # Simulation data (can be used for constructing tracks)
        with open("data_out/conflict_shav_{}_data_{}.json".format(wmspec,expstring), "w") as f:
            json.dump(data, f)

        # Simulation summary statistics.
        with open("data_out/conflict_shav_{}_stats_{}.json".format(wmspec,expstring), "w") as f:
            json.dump(stats, f)
