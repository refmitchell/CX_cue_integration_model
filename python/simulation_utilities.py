"""
simulation_utilities.py

This module provides the core behaviorual simulation described in the paper
(perform_runs()). There are subroutines for noisy dances and computing statistics
for each agent.
"""

import numpy as np
import test_utilities as testutils
from extended_ring_model import *
from dict_key_definitions import rmkeys, simkeys, statskeys, simdatakeys

from test_utilities import angular_difference, circ_scatter, circmean

# Define sane simulation defaults which may be modified
__def_s1 = 100
__def_s2 = 105
__def_sm_seed = 110
__def_placement_seed = 924
__def_desired_seed = 3432984

default_param_dict = {
    simkeys.c2_offsets:[], # Introduce cue offset/conflict (empty list or one per agent)
    simkeys.desired:[], # May want to inject a desired direction
    simkeys.k1:0, # Set to 0 to disable
    simkeys.k2:0, # Set to 0 to disable
    simkeys.w1:0.5, # w1, w2 should sum to 1
    simkeys.w2:0.5, # w1, w2 should sum to 1
    simkeys.s1:__def_s1, # Any integer
    simkeys.s2:__def_s2, # Any integer
    simkeys.timeout:100, # Positive integer
    simkeys.n_rolls:10, # Postive integer
    simkeys.max_turn:24, # Positive integer
    simkeys.step_size:2.89, # Included for completeness, shouldn't be modified
    simkeys.arena_size:30, # Positive integer
    simkeys.sm_seed:__def_sm_seed, # Integer
    simkeys.sm_noise:True, # Disables self-motion noise
    simkeys.flatten_on_first:True, # Clears snapshot on first roll
    simkeys.reseed_generators:False, # Systematically re-seeds generators on each roll
    simkeys.desired_angle_seed:__def_desired_seed, # Integer
    simkeys.placement_angle_seed:__def_placement_seed, # Integer
    simkeys.cue_one_noise_gen:np.random.RandomState(__def_s1),
    simkeys.cue_two_noise_gen:np.random.RandomState(__def_s2),
    simkeys.sm_noise_gen:np.random.RandomState(__def_sm_seed),
    simkeys.placement_gen:np.random.RandomState(__def_sm_seed),
    simkeys.desired_direction_gen:np.random.RandomState(__def_desired_seed)
}

def perform_runs(agents, params=dict()):
    # Unpack parameters - all default seeds are arbitrarily chosen
    c2_offsets = params.get(simkeys.c2_offsets, default_param_dict[simkeys.c2_offsets])
    desired_epg_presets = params.get(simkeys.desired, default_param_dict[simkeys.desired])
    k1 = params.get(simkeys.k1, default_param_dict[simkeys.k1])
    k2 = params.get(simkeys.k2, default_param_dict[simkeys.k2])
    w1 = params.get(simkeys.w1, default_param_dict[simkeys.w1])
    w2 = params.get(simkeys.w2, default_param_dict[simkeys.w2])
    s1 = params.get(simkeys.s1, default_param_dict[simkeys.s1])
    s2 = params.get(simkeys.s2, default_param_dict[simkeys.s2])
    timeout = params.get(simkeys.timeout, default_param_dict[simkeys.timeout])
    n_rolls = params.get(simkeys.n_rolls, default_param_dict[simkeys.n_rolls])
    max_turn = params.get(simkeys.max_turn, default_param_dict[simkeys.max_turn])
    step_size = params.get(simkeys.step_size, default_param_dict[simkeys.step_size])
    arena_size = params.get(simkeys.arena_size, default_param_dict[simkeys.arena_size])
    sm_seed = params.get(simkeys.sm_seed, default_param_dict[simkeys.sm_seed])
    sm_noise = params.get(simkeys.sm_noise, default_param_dict[simkeys.sm_noise])
    flatten_on_first = params.get(simkeys.flatten_on_first,
                                  default_param_dict[simkeys.flatten_on_first])
    desired_angle_seed = params.get(simkeys.desired_angle_seed,
                                    default_param_dict[simkeys.desired_angle_seed])
    placement_angle_seed = params.get(simkeys.placement_angle_seed,
                                      default_param_dict[simkeys.placement_angle_seed])
    reseed_generators = params.get(simkeys.reseed_generators,
                                   default_param_dict[simkeys.reseed_generators])

    # If RandomState objects are explicitly passed then seed settings are
    # implicitly ignored (no generator is ever initialised with passed seeds).
    # If no RandomState is passed, then new ones are instantiated using either
    # passed seeds or defaults.
    gen1 = params.get(simkeys.cue_one_noise_gen,
                      np.random.RandomState(s1))
    gen2 = params.get(simkeys.cue_two_noise_gen,
                      np.random.RandomState(s2))
    desired_gen = params.get(simkeys.desired_direction_gen,
                             np.random.RandomState(desired_angle_seed))
    placement_gen = params.get(simkeys.placement_gen,
                               np.random.RandomState(placement_angle_seed))
    sm_gen = params.get(simkeys.sm_noise_gen,
                        np.random.RandomState(sm_seed))

    # If desired directions specified
    # Check there are enough desired directions for each agent, do nothing if not
    if desired_epg_presets != []:
        if len(desired_epg_presets) != len(agents):
            print("Error: the number of desired EPG presets does not match the number of agents")
            return

    if c2_offsets != []:
        if len(c2_offsets) != len(agents):
            print("Error: Cue two offsets must match the number of agents being simulated.")
            return

    # Initialise data structures
    agent_data = []
    world_data = []
    snapshot_data = []

    for idx in range(len(agents)):
        a = agents[idx]
        agent_tracks = []
        world_tracks = []
        agent_snapshots = []

        desired_epg_angle = desired_gen.uniform(low=-np.pi, high=np.pi)
        if desired_epg_presets != []:
            desired_epg_angle = desired_epg_presets[idx]

        c2_offset = 0
        if c2_offsets != []:
            c2_offset = c2_offsets[idx]

        for run in range(n_rolls):
            t = 0
            dist = 0

            world_orientation = placement_gen.uniform(low=-np.pi, high=np.pi)
            cue_start = -np.degrees(world_orientation)
            c1 = cue_start
            c2 = cue_start - c2_offset

            # Hold in arena centre to allow the network time to stabilise
            for t in range(15):
                a.update_state(c1, c2, 0, w1=w1, w2=w2)

            # Only flatten weights if first run and we don't want to
            # carry over from previous simulation.
            flatten = True if (run == 0 and flatten_on_first) else False
            _, (c1, c2) = perform_snapshot(a,
                                           k1=k1,
                                           k2=k2,
                                           w1=w1,
                                           w2=w2,
                                           c1=c1,
                                           c2=c2,
                                           s1=s1,
                                           s2=s2,
                                           flatten=flatten)

            # Want to know both weight matrices and whether flattening occurred
            agent_snapshots.append((a.w_r1_epg, a.w_r2_epg, flatten))

            plot = False
            if plot:
                plt.subplot(121)
                plt.pcolormesh(a.w_r1_epg, vmax=0.25)
                plt.title("w1 = {}".format(w1))
                plt.subplot(122)
                plt.pcolormesh(a.w_r2_epg, vmax=0.25)
                plt.show()

            agent_position = (0, 0, 0, a.decode()[decodekeys.epg][0], desired_epg_angle)
            world_position = (0, 0, 0, world_orientation, desired_epg_angle)

            agent_track = [agent_position]
            world_track = [world_position]

            while (t < timeout) and (dist < arena_size):
                t += 1
                ss = step_size

                # Signed error between current E-PG angle and desired
                internal_angle, _ = a.decode()[decodekeys.epg]
                desired = [np.cos(desired_epg_angle), np.sin(desired_epg_angle)]
                internal = [np.cos(internal_angle), np.sin(internal_angle)]
                det = desired[0]*internal[1] - desired[1]*internal[0]
                dot = desired[0]*internal[0] + desired[1]*internal[1]
                error = np.degrees(np.arctan2(det,dot))

                # Turn + noise
                turn = 0
                if abs(error) <= max_turn:
                    turn = -error
                else:
                    ss = 0
                    turn = -max_turn if error > 0 else max_turn

                true_turn = turn

                if sm_noise:
                    # Rough (Khaldy et al., 2021)
                    sm_noise = sm_gen.randint(-30,31)
                    true_turn = turn + sm_noise

                # Update world orientation based on noisy correction
                world_orientation += np.radians(true_turn)

                # Generate cue positional noise
                c1_noise = 0 if k1 == 0 else gen1.vonmises(0, k1)
                c2_noise = 0 if k2 == 0 else gen2.vonmises(0, k2)

                # Cues move with the true motion of the agent
                c1 = np.degrees(world_orientation) + np.degrees(c1_noise)
                c2 = np.degrees(world_orientation) - c2_offset + np.degrees(c2_noise)

                # Agent update - agent perceives only the intended turn
                a.update_state(c1, c2, turn, w1=w1, w2=w2) # TEST
                # a.update_state(c1, c2, 0, w1=w1, w2=w2) # TEST

                # E-PGs and world position are set for this step
                # Agent positional update
                agent_angle, _ = a.decode()[decodekeys.epg]
                agent_cart_step = (ss*np.cos(agent_angle),
                                   ss*np.sin(agent_angle))
                agent_position = (agent_position[0] + agent_cart_step[0],
                                  agent_position[1] + agent_cart_step[1],
                                  ss,
                                  agent_angle,
                                  desired_epg_angle)

                agent_track.append(agent_position)

                # World positional update
                world_cart_step = (ss*np.cos(world_orientation),
                                   ss*np.sin(world_orientation))
                world_position = (world_position[0] + world_cart_step[0],
                                  world_position[1] + world_cart_step[1],
                                  ss,
                                  world_orientation,
                                  desired_epg_angle)
                world_track.append(world_position)

                # Distance from start position - world-based measure
                dist = np.sqrt((world_position[0] - 0)**2 + (world_position[1] - 0)**2)

            agent_tracks.append(agent_track)
            world_tracks.append(world_track)

            if reseed_generators:
                s1 += 1
                s2 += 1
                placement_angle_seed += 1
                gen1 = np.random.RandomState(s1)
                gen2 = np.random.RandomState(s2)
                placement_gen = np.random.RandomState(placement_seed)

        agent_data.append(agent_tracks)
        world_data.append(world_tracks)
        snapshot_data.append(agent_snapshots)

    data_dict = dict()
    data_dict[simdatakeys.agent] = agent_data
    data_dict[simdatakeys.world] = world_data
    data_dict[simdatakeys.snapshots] = snapshot_data

    # Usage:
    # Calling perform_runs will return a dictionary with three entries
    # agent - the agent's perceived movements
    # world - the agent's true movements in the world
    # snapshot - the snapshots taken on each roll.
    #
    # Data format:
    # World and agent behavioural data
    # [ # data
    #     [ # data[x] : all for an individual x
    #         [()], # data[x][y], individual x, roll y
    #         [()],
    #         [()]
    #     ]
    # ]
    #
    #
    # for individual in data:
    #     for roll in individual:
    #         for step in roll:
    #             step[0] = x
    #             step[1] = y
    #             step[2] = step size
    #             step[3] = angle in E-PGs for this step
    #             step[4] = desired epg angle

    # Snapshot data
    # data[x] = data for individual x
    # data[x][y] = snapshot data for roll y of individual x
    # data[x][y][a] -> a = 0, w_r1_epg; a = 1, w_r2_epg; a = 2, flatten (boolean)
    return data_dict

def initialise(n=1, d_w1=0.5, d_w2=0.5):
    agents = []

    for x in range(n):
        agents.append(RingModel({rmkeys.d_w1:d_w1, rmkeys.d_w2:d_w2}))

    for a in agents: # Init all agents
        a.initialise(w1=d_w1, w2=d_w2)

    return agents

def perform_snapshot(agent,
                     k1=0,
                     k2=0,
                     w1=0.5,
                     w2=0.5,
                     plasticity=True,
                     d=15,
                     s1=100,
                     s2=105,
                     c1=0,
                     c2=0,
                     flatten=True):
    # Random generators
    gen1 = np.random.RandomState(s1)
    gen2 = np.random.RandomState(s2)

    duration = d

    # Flatten adjacency matrices for learning
    n_r1 = agent.n_r1
    n_r2 = agent.n_r2
    base = 1/n_r1 if n_r1 > n_r2 else 1/n_r2

    # ON by default
    if flatten:
        agent.w_r1_epg = np.zeros((8,n_r1)) + base
        agent.w_r2_epg = np.zeros((8,n_r2)) + base

    for t in range(duration):
        angle_update = 24
        c1_noise = 0 if k1 == 0 else gen1.vonmises(0, k1)
        c2_noise = 0 if k2 == 0 else gen2.vonmises(0, k2)

        c1 += angle_update
        c2 += angle_update

        c1_sample = c1 + np.degrees(c1_noise)
        c2_sample = c2 + np.degrees(c2_noise)
        sm = angle_update

        agent.update_state(c1_sample,
                           c2_sample,
                           w1=w1,
                           w2=w2,
                           sm=sm,
                           plasticity=plasticity
        )

    return agent, (c1%360, c2%360)

def summary_statistics(dataset,
                       arena_size=default_param_dict[simkeys.arena_size],
                       step_size=default_param_dict[simkeys.step_size],
                       shav_exclusion=False
):
    # Compute mean vector for exit angles, compute total distance travelled,
    # Compute distribution of headings (this is probably the most important one).
    # dataset[x] - all runs for individual x
    # dataset[x][y] - Individual x, run y
    # dataset[x][y][z] - Individual x, run y, step z = (x,y,r,t,goal)

    # Results dictionary
    # "avg_distance" : The average distance travelled over all runs of an agent
    # "avg_time" : The average time taken to exit the arena
    # "exit_vectors" : A list of all n_rolls exit vectors; mag included to check for exit
    # "mean_vectors" : The mean polar vector for each agent
    # "overall_mean" : An overall mean vector (mean of "mean_vectors")
    # "overall_avg_dist" : The overall average distance travelled for this dataset
    # "head_directions" : A list of all head directions experienced by the agent

    results_dict = dict()
    average_dists = []
    agent_means = []
    exit_vectors = []
    head_directions = []
    step_lengths = []
    average_times = []
    exit_angles = []

    for agent in dataset:
        #
        # Average distance travelled by the agent over all runs
        #
        times = [ len(run) for run in agent ] # No. timesteps
        average_times.append(np.mean(times))

        #
        # Agent mean vector
        #
        final_coords = [ run[-1] for run in agent ]
        angles = [ np.arctan2(y,x) for (x,y,_,_,_) in final_coords ]
        displacements = [np.sqrt(x**2 + y**2) for (x,y,_,_,_) in final_coords]
        exits = []
        for (a,d) in zip(angles, displacements):
            if d >= arena_size:
                exits.append(a)
            else: # Modified to make sure there are always n exits.
                print("simutils: failed exit")
                exits.append(np.nan)
        exit_angles.append(exits)
        agent_means.append(circmean(angles))

        #
        # Exit vectors
        #
        exits = list(zip(displacements,angles))
        exit_vectors.append(exits)

        angles = []
        dists = []

        for run in agent:
            steps = [s for (_,_,s,_,_) in run]
            dists.append(sum(steps)) # Total distance travelled per run
            for (_,_,_,t,_) in run:
                angles.append(t)

        head_directions.append(angles)
        average_dists.append(np.mean(dists))

    overall_mean = circmean(
        [ t for (_, t) in agent_means ],
        weights=[ r for (r, _) in agent_means ]
    )

    # [(r,t,k)], t being the average E-PG angle, r being the mean vector length
    # k is kappa for von Mises distribution (assumed)
    mean_headings = [
        circmean(x,include_kappa=True)
        for x in head_directions
    ]

    # One result per agent
    results_dict[statskeys.avg_distance] = average_dists
    results_dict[statskeys.avg_time] = average_times
    results_dict[statskeys.exit_vectors] = exit_vectors
    results_dict[statskeys.mean_vectors] = agent_means
    results_dict[statskeys.head_directions] = head_directions
    results_dict[statskeys.mean_heading_vectors] = mean_headings
    results_dict[statskeys.exit_angles] = exit_angles

    # One result per dataset
    results_dict[statskeys.overall_mean] = overall_mean
    results_dict[statskeys.overall_avg_dist] = np.mean(average_dists)

    return results_dict

