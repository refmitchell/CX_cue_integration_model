"""
[Legacy] - This is a very old version of the simulation routine in
simulation_utilities.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from extended_ring_model import *
from test_utilities import angular_difference, circ_scatter
import sys
#from scipy.stats import vonmises
import copy

class Agent():
    def __init__(self):
        self.model = RingModel()
        self.model.initialise()
        self.true_position = (0, 0, 0, a.decode()["epg"][0],desired_epg_angle)
        self.epg_position = (0, 0, 0, a.decode()["epg"][0],desired_epg_angle)

def circmean(angles, weights=None, scale=1,include_kappa=False):
    print(arena_size)
    if len(angles) == 0:
        return (0,0)
    if weights != None:
        if len(weights) != len(angles):
            print("Err: weights did not match angles, using equal weights")
            weights = [ 1 for x in angles ]

    if weights == None:
        weights = [ 1 for x in angles ]

    weights = [ arena_size if x > arena_size else x for x in weights]

    cartesian =[ (r*np.cos(t), r*np.sin(t)) for (t,r) in zip(angles,weights)]
    xs = [ x for (x,_) in cartesian ]
    ys = [ y for (_,y) in cartesian ]
    avg_x = sum(xs)/len(xs)
    avg_y = sum(ys)/len(ys)

    if include_kappa:
        R = scale*np.sqrt(avg_x**2 + avg_y**2)
        kappa = 0
        # Mardia and Jupp (2009) - Kappa approximation
        if R < 0.53:
            kappa = 2*R + R**3 + (5/6)*(R**5)
        elif R >= 0.85:
            kappa = 1 / (2*(1 - R) - (1-R)**2 - (1 - R)**3)
        else:
            kappa = -0.4 + 1.39*R + (0.43 / (1-R))
        mean = (R*scale, np.arctan2(avg_y, avg_x), kappa)
        return mean

    mean = (scale*np.sqrt(avg_x**2 + avg_y**2), np.arctan2(avg_y, avg_x))
    return mean

def perform_runs(agents, k1, k2, w1, w2, s1, s2,
                 timeout=5000,
                 n_rolls=10,
                 max_turn=24,#20,
                 step_size=2.89,
                 arena_size=30,
                 sm_seed=54138641
):
    # Seed random generators
    gen1 = np.random.RandomState(s1)
    gen2 = np.random.RandomState(s2)

    # Check it works before including these
    desired_epg_angle_seed = 924
    placement_seed = 3432984
    desired_gen = np.random.RandomState(desired_epg_angle_seed)
    placement_gen = np.random.RandomState(placement_seed)
    sm_gen = np.random.RandomState(sm_seed)
    sm_noise = sm_seed != None

    agent_data = []
    world_data = []

    counter = 0


    for a in agents:
        agent_tracks = []
        world_tracks = []

        desired_epg_angle = desired_gen.uniform(low=-np.pi, high=np.pi)

        debug_perceived = []
        debug_actual = []

        for run in range(n_rolls):
            t = 0
            dist = 0

            world_orientation = placement_gen.uniform(low=-np.pi, high=np.pi)
            cue_start = -np.degrees(world_orientation)
            c1 = cue_start
            c2 = cue_start
            # if run >= 3:
            #     c2 = cue_start + 180

            # Hold in arena centre to allow the network time to stabilise
            for t in range(15):
                a.update_state(c1, c2, 0, w1=w1, w2=w2)

            # plt.subplot(121)
            # plt.pcolormesh(a.w_r1_epg, vmax=0.25)
            # plt.subplot(122)
            # plt.pcolormesh(a.w_r2_epg, vmax=0.25)
            # plt.show()


            plot=False
            if w1 != 1 and plot:
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5,2.5))
                axs[0].set_title("w1 = {}".format(w1))
                axs[0].pcolormesh(a.w_r1_epg, vmax=0.5)
                axs[1].pcolormesh(a.w_r2_epg, vmax=0.5)
                plt.show()

            # Test: Dancing on each arena placement. Snapshots
            _, (c1, c2) = perform_snapshot(a,
                                           k1=k1,
                                           k2=k1,
                                           w1=w1,
                                           w2=w2,
                                           c1=c1,
                                           c2=c2,
                                           s1=s1,
                                           s2=s2,
                                           flatten=False)



            agent_position = (0,0,0, a.decode()["epg"][0],desired_epg_angle)
            world_position = (0,0,0, world_orientation,desired_epg_angle)

            agent_track = [agent_position]
            world_track = [world_position]

            # print("New run: {} - w1 = {}, w2 = {}".format(run, w1, w2))
            while (t < timeout) and (dist < arena_size):
                t += 1
                ss = step_size

                # Signed error between current E-PG angle and desired
                internal_angle, _ = a.decode()["epg"]
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
                    sm_noise = sm_gen.randint(-30,31) # Rough (Khaldy et al., 2021)
                    true_turn = turn + sm_noise

                # Update world orientation based on noisy correction
                world_orientation += np.radians(true_turn)

                # Generate cue positional noise
                c1_noise = 0 if k1 == 0 else gen1.vonmises(0, k1)
                c2_noise = 0 if k2 == 0 else gen2.vonmises(0, k2)

                # Cues move with the true motion of the agent
                c1 = np.degrees(world_orientation) + np.degrees(c1_noise)
                c2 = np.degrees(world_orientation) + np.degrees(c2_noise)

                # Agent update - agent perceives only the intended turn
                a.update_state(c1, c2, turn, w1=w1, w2=w2) # TEST
                #a.update_state(c1, c2, 0, w1=w1, w2=w2) # SM OFF
                #a.update_state(c1, c2, turn, w1=0, w2=0) # SM ONLY

#                print("EPG: {}".format(a.epg_rates))

                # E-PGs and world position are set for this step
                # Agent positional update
                agent_angle, _ = a.decode()["epg"]
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

                # Debug
#                perceived = np.degrees(angular_difference(agent_angle, internal_angle,signed=True))
                perceived = np.degrees(angular_difference(internal_angle, agent_angle,signed=True))
                debug_perceived.append(np.degrees(agent_angle) % 360)
                debug_actual.append(np.degrees(world_orientation) % 360)
                # print("{}:{}".format(perceived,true_turn))

            agent_tracks.append(agent_track)
            world_tracks.append(world_track)

            # Re-seed random generators with new (systematic) seed
            # If you have the same number of agents doing the same
            # number of rolls, then the noise experienced by agent x
            # in roll y of a given condition should be the same.
            s1 += 1
            s2 += 1
            placement_seed += 1
            gen1 = np.random.RandomState(s1)
            gen2 = np.random.RandomState(s2)
            placement_gen = np.random.RandomState(placement_seed)

        agent_data.append(agent_tracks)
        world_data.append(world_tracks)
        counter+=1
        debug_diffs = [np.abs(x - y) for (x,y) in zip(debug_perceived, debug_actual)]
        # DEBUG PLOTTING

#         fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8,4))

#         plt.plot(debug_perceived, label="Internal", marker='.')
#         plt.plot(debug_actual, label="Actual",marker='.')
# #        ax.plot(debug_diffs)
#         plt.legend()
# #        plt.ylim(-180,180)
#         plt.show()

    # Data format
    #
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
    return agent_data, world_data

#
# Wrappers
#
def single_cue(agents,
               kappa,
               seed=100,
               timeout=100,
               n_rolls=10,
               snapshot=True,
               arena_size=30):
    """
    Perform runs for the single cue case. As there is only one cue
    (c1), the weight (w1) is 1 due to our constraint that the weights
    of all available cues sum to 1.
    :param agents: A list of agents.
    :param kappa: The noise parameter for the cue.
    :param seed: The seed for the random generator to ensure consistency
                 between the test conditions.
    :return: 2D array of results. Each row is a trace.
    """
    k1 = kappa
    k2 = 0
    w1 = 1
    w2 = 0

    if snapshot:
        agents = perform_snapshots(agents,
                                   k1=k1, k2=k2,
                                   w1=w1, w2=w2,
                                   s1=s1, s2=s2)

    return perform_runs(agents,
                        kappa,
                        0, # kappa 2, irrelevant as w2 = 0
                        w1,
                        0,    # w2
                        seed,
                        0, # Unused as w2 = 0, needs to be a float
                        timeout=timeout,
                        n_rolls=n_rolls,
                        arena_size=arena_size)

def weight_by_reliability(agents,
                          k1,
                          k2,
                          s1=100,
                          s2=105,
                          timeout=100,
                          n_rolls=10,
                          snapshot=True,
                          arena_size=30):
    # Normalised weights, computed from reliability
    w1 = 0
    w2 = 0

    if k1 == 0 and k2 == 0:
        w1 = 0.5
        w2 = 0.5
    else:
        w1 = k1 / (k1 + k2)
        w2 = k2 / (k1 + k2)

    # if w1 > 0.5:
    #     w1 = 0.5
    #     w2 = w1 / (w1/w2)

    if snapshot:
        agents = perform_snapshots(agents,
                                   k1=k1, k2=k2,
                                   w1=w1, w2=w2,
                                   s1=s1, s2=s2)

    # a = agents[0]
    # plt.subplot(121)
    # plt.pcolormesh(a.w_r1_epg, vmax=1, vmin=0)
    # plt.subplot(122)
    # plt.pcolormesh(a.w_r2_epg, vmax=1, vmin=0)
    # plt.show()
    return perform_runs(agents,
                        k1,
                        k2,
                        w1,
                        w2,
                        s1,
                        s2,
                        timeout=timeout,
                        n_rolls=n_rolls,
                        arena_size=arena_size)

def weight_by_saliency(agents,
                       w1,
                       w2,
                       k1,
                       k2,
                       s1=100,
                       s2=105,
                       timeout=100,
                       n_rolls=10,
                       snapshot=True,
                       arena_size=30):
    if snapshot:
        agents = perform_snapshots(agents,
                                   k1=k1, k2=k2,
                                   w1=w1, w2=w2,
                                   s1=s1, s2=s2)


    return perform_runs(agents, k1, k2, w1, w2, s1, s2,
                        timeout=timeout, n_rolls=n_rolls, arena_size=arena_size)

def initialise(n=1, d_w1=0.5, d_w2=0.5):
    agents = []

    for x in range(n):
        agents.append(RingModel({"d_w1":d_w1, "d_w2":d_w2}))

    for a in agents: # Init all agents
        a.initialise(w1=d_w1, w2=d_w2)

    return agents

def perform_snapshots(agents,
                      k1=0,
                      k2=0,
                      w1=0.5,
                      w2=0.5,
                      d=15,
                      s1=100,
                      s2=105):

    for idx in range(len(agents)):
        agents[idx], _ = perform_snapshot(agents[idx],
                                       k1=k1,
                                       k2=k2,
                                       w1=w1,
                                       w2=w2,
                                       d=d,
                                       s1=s1,
                                       s2=s2)
        # Update the seeds. Each agent should experience the
        # same noise per run, but different agents should get
        # different noise sequences.
        s1 += 1
        s2 += 1

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
        # print(np.degrees(agent.decode()["epg"][0]))
    #print("c1 dance: {}".format(c2 % 360))

    return agent, (c1%360, c2%360)

def summary_statistics(dataset, arena_size=50, step_size=2.89):
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
        exit_angles.append(exits)
        agent_means.append(circmean(angles, weights=displacements))

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
        weights=[ r for (r, _) in agent_means ])

    # [(r,t,k)], t being the average E-PG angle, r being the mean vector length
    # k is kappa for von Mises distribution (assumed)
    mean_headings = [ circmean(x,include_kappa=True) for x in head_directions ]

    # One result per agent
    results_dict["avg_distance"] = average_dists
    results_dict["avg_time"] = average_times
    results_dict["exit_vectors"] = exit_vectors
    results_dict["mean_vectors"] = agent_means
    results_dict["head_directions"] = head_directions
    results_dict["mean_heading_vectors"] = mean_headings
    results_dict["exit_angles"] = exit_angles

    # One result per dataset
    results_dict["overall_mean"] = overall_mean
    results_dict["overall_avg_dist"] = np.mean(average_dists)

    return results_dict

if __name__ == "__main__":
    # 'Primary' cue, we add a less reliable one.
    # Weight-by-reliability says the agent will get more precise.
    # Weight-by-saliency says the agent's precision depends on
    # the saliency of the 'strong' cue. Cues are assumed aligned
    # and positioned at 0.

    # Seeds: the noise from each cue should be consistently applied
    # so that differences arise only from the 'strategy' used to
    # combine the cues. Currently random for testing purposes.
    s1 = np.random.randint(0, high=100000000) # 450
    s2 = np.random.randint(0, high=100000000) # 120

    # s1 = 22631318
    # s2 = 75375299

    # s1 = 35977264
    # s2 = 80473687

    print("S1: {}".format(s1))
    print("S2: {}".format(s2))

    # Kappas: the 'reliability' of each cue
    k1 = 2
    # k1 = 1.25
    k2 = 0.5

    # Weights: the 'saliancy' of each cue
    w1 = 0.2
    w2 = 0.8

    n = 10 # Number of agents
    n_rolls = 10 # Number of 'rolls' per agent

    # I don't know why this deepcopying is necessary.
    # Copies have to be assigned for some reason?
    agents = initialise(n=n, d_w1=1, d_w2=0)
    agents2 = initialise(n=n, d_w1=k1/(k1+k2), d_w2=k2/(k1+k2)) # copy.deepcopy(agents)
    agents3 = initialise(n=n, d_w1=w1, d_w2=w2) # copy.deepcopy(agents)
    print("\n====INIT COMPLETE====\n")
    # Misc. simualtion properties
    timeout = 100
    snapshot = True
    arena_size = 30

    sng_agent, sng_world = single_cue(agents,
                                      k1,
                                      seed=s1,
                                      n_rolls=n_rolls,
                                      snapshot=snapshot,
                                      timeout=timeout,
                                      arena_size=arena_size)

    rel_agent, rel_world = weight_by_reliability(agents2, k1, k2,
                                                 s1=s1, s2=s2, n_rolls=n_rolls,snapshot=snapshot,
                                                 timeout=timeout, arena_size=arena_size)
    sal_agent, sal_world = weight_by_saliency(agents3, w1, w2, k1, k2,
                                              s1=s1, s2=s2, n_rolls=n_rolls, snapshot=snapshot,
                                              timeout=timeout, arena_size=arena_size)


    single_stats = summary_statistics(sng_agent, arena_size=arena_size)
    rel_stats = summary_statistics(rel_agent, arena_size=arena_size)
    sal_stats = summary_statistics(sal_agent, arena_size=arena_size)

    sng_world_stats = summary_statistics(sng_world, arena_size=arena_size)
    rel_world_stats = summary_statistics(rel_world, arena_size=arena_size)
    sal_world_stats = summary_statistics(sal_world, arena_size=arena_size)

    single_kappa = np.mean([x[2] for x in single_stats["mean_heading_vectors"]])
    rel_kappa = np.mean([x[2] for x in rel_stats["mean_heading_vectors"]])
    sal_kappa = np.mean([x[2] for x in sal_stats["mean_heading_vectors"]])

    # print("A-SNG: {:.1f}".format(single_kappa))
    # print("A-REL: {:.1f}".format(rel_kappa))
    # print("A-SAL: {:.1f}".format(sal_kappa))

    sng_world_kappa = np.mean([x[2] for x in sng_world_stats["mean_heading_vectors"]])
    rel_world_kappa = np.mean([x[2] for x in rel_world_stats["mean_heading_vectors"]])
    sal_world_kappa = np.mean([x[2] for x in sal_world_stats["mean_heading_vectors"]])
    # print("W-SNG: {:.1f}".format(sng_world_kappa))
    # print("W-REL: {:.1f}".format(rel_world_kappa))
    # print("W-SAL: {:.1f}".format(sal_world_kappa))

    #
    # Plotting
    #
    nrows = min(3,n)

    mosaic = []
    for idx in range(nrows):
        mosaic_idx = idx+1
        mosaic_row = ["{}", "{}r", "{}s"] # [x, x reliability, x saliency]
        mosaic_row = [ e.format(mosaic_idx) for e in mosaic_row ]
        mosaic.append(mosaic_row)

#     fig, axs = plt.subplot_mosaic(mosaic,
#                                   figsize=(8,nrows*3),
#                                   sharey=True,
#                                   subplot_kw={"projection":"polar"})

#     fig.tight_layout()

#     for individual in range(nrows):
#         rowkey = str(individual + 1)

#         # Parallel lists
#         d = [sng_agent[individual], rel_agent[individual], sal_agent[individual]]
#         axkeys = [rowkey, rowkey+"r", rowkey+"s"]
#         stats_dicts = [single_stats, rel_stats, sal_stats]
#         titles = ["a) Single Cue", "b) Reliability", "c) Strength"]
#         for idx in range(len(d)):
#             condition = d[idx]
#             axkey = axkeys[idx]

#             if individual == 0:
#                 axs[axkey].set_title(titles[idx])

#             axs[axkey].set_theta_direction(-1)
#             axs[axkey].set_theta_zero_location("N")
#             axs[axkey].set_yticks([])
#             axs[axkey].set_ylim([0,arena_size + 3])
#             axs[axkey].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])

#             T = stats_dicts[idx]["mean_vectors"][individual][1]
#             R = stats_dicts[idx]["mean_vectors"][individual][0]
#             axs[axkey].annotate("",
#                                 xy=(T,R),
#                                 xytext=(0,0),
#                                 arrowprops=dict(arrowstyle="-|>", fill=True, color='k')

#             )
#             # axs[axkey].text(np.pi, 85,
#             #                 "R: {:.1f}, $\\theta$: {:.1f}$^\degree$, $\mu_D$: {:.1f}cm".format(R, np.degrees(T) % 360, stats_dicts[idx]["avg_distance"][individual]),
#             #                 horizontalalignment='center',
#             #                 bbox=dict(facecolor='1', edgecolor='grey', pad=2.0)
#             # )
#             axs[axkey].text(np.pi, arena_size + 25,
#                             "$\mu_D$: {:.1f}cm, T = {:.1f}t, R = {:.1f}".format(stats_dicts[idx]["avg_distance"][individual], stats_dicts[idx]["avg_time"][individual], stats_dicts[idx]["mean_vectors"][individual][0]),
#                             horizontalalignment='center'
# #                            bbox=dict(facecolor='',  pad=2.0)
#             )
#             for run in condition:
#                 ts = [ np.arctan2(y,x) for (x,y,_,_,_) in run ]
#                 rs = [ np.sqrt(x**2 + y**2) for (x,y,_,_,_) in run ]
#                 axs[axkey].plot(ts, rs, alpha=0.8)#, color='tab:blue')

    #############################################################################
    # WORLD TRACKS                                                              #
    #############################################################################
    fig, axs = plt.subplot_mosaic(mosaic,
                                  figsize=(4,nrows*2),
                                  sharey=True,
                                  subplot_kw={"projection":"polar"})

    fig.tight_layout()

    for individual in range(nrows):
        rowkey = str(individual + 1)

        # Parallel lists
        d = [sng_world[individual], rel_world[individual], sal_world[individual]]
        axkeys = [rowkey, rowkey+"r", rowkey+"s"]
        stats_dicts = [sng_world_stats, rel_world_stats, sal_world_stats]
        titles = ["a) Single Cue", "b) Reliability", "c) Strength"]
        for idx in range(len(d)):
            condition = d[idx]
            axkey = axkeys[idx]

            if individual == 0:
                axs[axkey].set_title(titles[idx])

            axs[axkey].set_theta_direction(-1)
            axs[axkey].set_theta_zero_location("N")
            axs[axkey].set_yticks([])
            axs[axkey].set_ylim([0,arena_size + 3])
            axs[axkey].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])

            T = stats_dicts[idx]["mean_vectors"][individual][1]
            R = stats_dicts[idx]["mean_vectors"][individual][0]
            axs[axkey].annotate("",
                                xy=(T,R),
                                xytext=(0,0),
                                arrowprops=dict(arrowstyle="-|>", fill=True, color='k')

            )
            # axs[axkey].text(np.pi, 85,
            #                 "R: {:.1f}, $\\theta$: {:.1f}$^\degree$, $\mu_D$: {:.1f}cm".format(R, np.degrees(T) % 360, stats_dicts[idx]["avg_distance"][individual]),
            #                 horizontalalignment='center',
            #                 bbox=dict(facecolor='1', edgecolor='grey', pad=2.0)
            # )
            axs[axkey].text(np.pi, arena_size + 25,
                            "$\mu_D$: {:.1f}cm, T = {:.1f}t, R = {:.1f}".format(stats_dicts[idx]["avg_distance"][individual], stats_dicts[idx]["avg_time"][individual], stats_dicts[idx]["mean_vectors"][individual][0]),
                            horizontalalignment='center'
#                            bbox=dict(facecolor='',  pad=2.0)
            )
            for run in condition:
                ts = [ np.arctan2(y,x) for (x,y,_,_,_) in run ]
                rs = [ np.sqrt(x**2 + y**2) for (x,y,_,_,_) in run ]
                axs[axkey].plot(ts, rs, alpha=0.8)#, color='tab:blue')

    plt.savefig("plots/traces.pdf", bbox_inches="tight")

    # #
    # # DISTRIBUTION PLOTS
    # #
    # fig, axs = plt.subplot_mosaic(mosaic,
    #                               figsize=(8,nrows*(8/nrows)),
    #                               sharey=True,
    #                               sharex=True)

    # fig.tight_layout()
    # plt.subplots_adjust(hspace=0.1, wspace=0.1)
    # for individual in range(nrows):
    #     rowkey = str(individual + 1)

    #     # Parallel lists
    #     d = [single[individual], rel[individual], sal[individual]]
    #     axkeys = [rowkey, rowkey+"r", rowkey+"s"]
    #     stats_dicts = [single_stats, rel_stats, sal_stats]
    #     titles = ["a) Single Cue", "b) Reliability", "c) Saliency"]

    #     for idx in range(len(d)):
    #         condition = d[idx]
    #         axkey = axkeys[idx]
    #         stats = stats_dicts[idx]

    #         if individual == 0:
    #             axs[axkey].set_title(titles[idx])

    #         headings = stats["head_directions"][individual]

    #         R = stats["mean_heading_vectors"][individual][0]
    #         T = stats["mean_heading_vectors"][individual][1]
    #         kappa = stats["mean_heading_vectors"][individual][2]

    #         x = np.linspace(-np.pi, np.pi, 1000)
    #         ys = vonmises.pdf(x, kappa, loc=T)

    #         axs[axkey].plot(x, ys, color='k')
    #         axs[axkey].hist(headings,
    #                         bins=72,
    #                         range=(-np.pi, np.pi),
    #                         density=True,
    #                         color='grey')


    #         ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    #         axs[axkey].set_xticks(ticks)
    #         axs[axkey].set_xticklabels(["{}$^\degree$".format(int(np.degrees(x))) for x in ticks])

    # for individual in range(nrows):
    #     # Dupe loop is so that the axes limits are set before drawing
    #     # this text box. Not nice but it works.
    #     rowkey = str(individual + 1)
    #     d = [single[individual], rel[individual], sal[individual]]
    #     axkeys = [rowkey, rowkey+"r", rowkey+"s"]
    #     stats_dicts = [single_stats, rel_stats, sal_stats]
    #     for idx in range(len(d)):
    #         condition = d[idx]
    #         axkey = axkeys[idx]
    #         stats = stats_dicts[idx]
    #         headings = stats["head_directions"][individual]
    #         R = stats["mean_heading_vectors"][individual][0]
    #         T = stats["mean_heading_vectors"][individual][1]
    #         kappa = stats["mean_heading_vectors"][individual][2]
    #         axs[axkey].text(0,
    #                         axs[axkey].get_ylim()[1] - 0.2,
    #                         "$n$ = {}, $\kappa$ = {:.1f}, $\mu$ = {:.1f}$^\degree$".format(
    #                             len(headings),
    #                             kappa,
    #                             np.degrees(T) % 360
    #                         ),
    #                         ha='center',
    #                         va='center',
    #                         bbox=dict(facecolor='1', edgecolor='grey', pad=1.5)
    #         )

    # # plt.savefig("plots/behaviour.pdf", bbox_inches="tight")

    #
    # EXIT ANGLE PLOTS
    #


    # fig, axs = plt.subplot_mosaic(mosaic,
    #                               figsize=(8,nrows*3),
    #                               sharey=True,
    #                               subplot_kw={"projection":"polar"})

    # fig.tight_layout()
    # rlim=1.5
    # for individual in range(nrows):
    #     rowkey = str(individual + 1)

    #     # Parallel lists
    #     d = [sng_world[individual], rel_world[individual], sal_world[individual]]
    #     axkeys = [rowkey, rowkey+"r", rowkey+"s"]
    #     stats_dicts = [sng_world_stats, rel_world_stats, sal_world_stats]
    #     titles = ["a) Single Cue", "b) Reliability", "c) Strength"]
    #     for idx in range(len(d)):
    #         condition = d[idx]
    #         axkey = axkeys[idx]

    #         if individual == 0:
    #             axs[axkey].set_title(titles[idx])

    #         axs[axkey].set_theta_direction(-1)
    #         axs[axkey].set_theta_zero_location("N")
    #         axs[axkey].set_yticks([])
    #         axs[axkey].grid(False)
    #         axs[axkey].set_theta_zero_location("N")
    #         axs[axkey].set_theta_direction(-1)
    #         axs[axkey].set_ylim([0, rlim])
    #         axs[axkey].axis('off')

    #         fake_ax_n = 1000
    #         fake_ax_t = np.linspace(-np.pi, np.pi, fake_ax_n)
    #         fake_ax_r = np.ones((fake_ax_n,))
    #         axs[axkey].plot(fake_ax_t, fake_ax_r, color='k', linewidth=1)
    #         inset = 0.95
    #         axs[axkey].text(0, inset, "$0^\degree$", ha='center', va='top')
    #         axs[axkey].text(np.pi, inset, "$180^\degree$", ha='center', va='bottom')
    #         axs[axkey].text(np.pi/2, inset, "$90^\degree$", ha='right', va='center')
    #         axs[axkey].text(-np.pi/2, inset, "$-90^\degree$", ha='left', va='center')

    #         # Exits rounded to nearest five degrees
    #         exits = stats_dicts[idx]["exit_angles"][individual]
    #         exits = np.radians(np.around(np.degrees(exits)/5, decimals=0)*5)

    #         # Todo: Plot means
    #         # R,T = circmean(exits)
    #         # axs[axkey].annotate("",
    #         #                     xy=(T,R),
    #         #                     xytext=(0,0),
    #         #                     arrowprops=dict(arrowstyle="-|>", fill=True, color='k')

    #         # )
    #         # axs[axkey].text()

    #         radii, angles = circ_scatter(exits, radial_base=1.1)
    #         axs[axkey].scatter(angles, radii, color='tab:blue',
    #                            alpha=0.5, edgecolors='k', s=2)

    #############################################################################
    # Precision boxplots                                                        #
    #############################################################################
    individual = 0
    fig, ax = plt.subplots(ncols=1, nrows=1, sharey=True, figsize=(5,2.5))

    sng_exits = sng_world_stats["exit_angles"]
    rel_exits = rel_world_stats["exit_angles"]
    sal_exits = sal_world_stats["exit_angles"]

    # Cannot guarantee that all runs have the same length
    for idx in range(n):
        sng_exits[idx] = np.radians(np.around(np.degrees(sng_exits[idx])/5, decimals=0)*5)
        rel_exits[idx] = np.radians(np.around(np.degrees(rel_exits[idx])/5, decimals=0)*5)
        sal_exits[idx] = np.radians(np.around(np.degrees(sal_exits[idx])/5, decimals=0)*5)

    sng_mvls = [circmean(x)[0] for x in sng_exits]
    rel_mvls = [circmean(x)[0] for x in rel_exits]
    sal_mvls = [circmean(x)[0] for x in sal_exits]

    boxcolour='k'
    ax.boxplot([sng_mvls, rel_mvls, sal_mvls],
               showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor='silver', color=boxcolour),
               capprops=dict(color=boxcolour),
               whiskerprops=dict(color=boxcolour),
               medianprops=dict(color=boxcolour),
               zorder=0)
    a = 0.5
    jit = 0
    ax.scatter([1 + np.random.uniform(low=-jit, high=jit) for x in sng_mvls],sng_mvls, color='tab:blue', alpha=a)
    ax.scatter([2 + np.random.uniform(low=-jit, high=jit) for x in rel_mvls],rel_mvls, color='tab:blue', alpha=a)
    ax.scatter([3 + np.random.uniform(low=-jit, high=jit) for x in sal_mvls],sal_mvls, color='tab:blue', alpha=a)
#    ax.text(0.6, 0.1,"$n$ = {}".format(n), ha="left", va="bottom")
    ax.set_ylim([0,1])
    ax.set_ylabel("Average mean vector length (R)")
    ax.set_xticklabels(["Single Cue", "Reliability", "Strength"])
    ax.set_title("Mean precision in each condition ($n$ = {})".format(n))

#    fig.savefig("plots/precision_boxplot.svg", bbox_inches='tight')
    fig.savefig("plots/precision_boxplot.pdf", bbox_inches='tight')
    plt.show()

    #############################################################################
    # ICN FIGURE                                                                #
    #############################################################################
    fig, trace = plt.subplots(nrows=1,
                              ncols=1,
                              figsize=(4,4),
                              subplot_kw={"projection":"polar"})
    trace.set_theta_direction(-1)
    trace.set_theta_zero_location("N")
    trace.set_yticks([])
    trace.set_ylim([0,arena_size])
    trace.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])

    # trace.text(np.pi, arena_size + 25,
    #            "$\mu_D$: {:.1f}cm, T = {:.1f}t, R = {:.1f}".format(sng_world_stats["avg_distance"][individual], sng_world_stats["avg_time"][individual], sng_world_stats["mean_vectors"][individual][0]),
    #            horizontalalignment='center'
    # )

    for run in sng_world[individual]:
        ts = [ np.arctan2(y,x) for (x,y,_,_,_) in run ]
        rs = [ np.sqrt(x**2 + y**2) for (x,y,_,_,_) in run ]
        trace.plot(ts, rs, alpha=0.8)
    fig.savefig("plots/ex_trace.svg", bbox_inches="tight")

    fig, ex = plt.subplots(nrows=1,
                           ncols=1,
                           figsize=(4,4),
                           subplot_kw={"projection":"polar"})

    rlim=1.2
    fake_ax_n = 1000
    fake_ax_t = np.linspace(-np.pi, np.pi, fake_ax_n)
    fake_ax_r = np.ones((fake_ax_n,))
    inset = 0.95
    ex.plot(fake_ax_t, fake_ax_r, color='k', linewidth=1)
    ex.text(0, inset, "$0^\degree$", ha='center', va='top')
    ex.text(np.pi, inset, "$180^\degree$", ha='center', va='bottom')
    ex.text(np.pi/2, inset, "$90^\degree$", ha='right', va='center')
    ex.text(-np.pi/2, inset, "$270^\degree$", ha='left', va='center')

    # Exits rounded to nearest five degrees
    exits = sng_world_stats["exit_angles"][individual]
    exits = np.radians(np.around(np.degrees(exits)/5, decimals=0)*5)
    radii, angles = circ_scatter(exits, radial_base=1.1)
    ex.scatter(angles, radii, color='tab:blue',
                       alpha=0.9, edgecolors='k', s=70)
    R,T = circmean(exits)

    #Prettifying
    ex.text(0,0.4,"R = {:.2f}\n $\\theta = {:.1f}^\degree$".format(R,np.degrees(T)%360),
            ha="center", va="bottom", bbox=dict(facecolor='1', edgecolor='grey', pad=1.5)
    )
    ex.annotate("",
                xy=(T,R),
                xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", fill=True, color='k')

    )
    ex.vlines(0,ymin=0,ymax=1,color='tab:grey', linestyle='dashed', alpha=0.7)
    ex.scatter(0,0,color='k', s=10,zorder=100)

    big_t = T + 2*np.pi
    theta_arc = np.linspace(0,big_t,1000)
    ex.plot(theta_arc, [0.3 for x in theta_arc], color='tab:grey', alpha=0.7,linestyle='dashed')
    ex.text(big_t/2, 0.1, "$\\theta$", color="tab:grey", ha='center', va='center')
    ex.text(big_t-np.radians(20), 0.75*R, "$\\bar \\mu$, $R = | \\bar \\mu |$", color="tab:grey", ha='center', va='center')
    ex.set_theta_direction(-1)
    ex.set_theta_zero_location("N")
    ex.set_yticks([])
    ex.grid(False)
    ex.set_theta_zero_location("N")
    ex.set_theta_direction(-1)
    ex.set_ylim([0, rlim])
    ex.axis('off')
    fig.savefig("plots/ex_exits.svg", bbox_inches="tight")
    #plt.show()

