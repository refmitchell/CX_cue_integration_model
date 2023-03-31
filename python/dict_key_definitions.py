"""
dict_key_definitions.py

Dictionaries are used to define simulations and models.  Keys are
defined here as 'constants' so that keys can be consistently used
without fear of an incorrect keys creating silent bugs where default
values are used.

Only keys are specified, default values should be set by the relevant
script where required.

Each class specifies a set of keys for some datastructure. The module
creates instances of each of these classes which can be imported
by any other module which needs them.
"""

"""
Keys used for the data and stats dictionaries in precision.py
(and the related plotting file, plot_precision.py).
"""
class constkeys_precision_experiment_result_dict():
    __slots__ = ()
    exp1 = "exp1"
    exp2 = "exp2"
    exp3 = "exp3"
    exp4 = "exp4"
    exp5 = "exp5"
    exp6 = "exp6"
    exp7 = "exp7"
    exp8 = "exp8"
    initial = "initial"
    test = "test"

    def expkeys(self):
        return [
            self.exp1,
            self.exp2,
            self.exp3,
            self.exp4,
            self.exp5,
            self.exp6,
            self.exp7,
            self.exp8
        ]

    def runkeys(self):
        return [self.initial, self.test]

"""
Keys used for the data and stats dictionaries in modality_transfer.py
"""
class constkeys_modality_transfer_experiment_result_dict():
    __slots__ = ()
    roll1 = "roll1"
    roll2 = "roll2"
    roll3 = "roll3"
    roll4 = "roll4"

    def rollkeys(self):
        return [
            self.roll1,
            self.roll2,
            self.roll3,
            self.roll4
        ]

"""
Keys used for the data and stats dictionaries in conflict_weighting.py
"""
class constkeys_conflict_weight_result_dict():
    __slots__ = ()
    phase1 = "phase1"
    phase2 = "phase2"
    phase3 = "phase3"
    phase4 = "phase4"
    def rollkeys(self):
        return [
            self.phase1,
            self.phase2,
            self.phase3,
            self.phase4
        ]


"""
Keys used for the Dacke et al. (2019) conflict experimental phases.
"""
class constkeys_conflict_dacke_result_dict():
    __slots__ = ()
    phase1 = "phase1" # Pre-pre-conflict rolls
    phase2 = "phase2" # Pre-conflict roll
    phase3 = "phase3" # Conflict roll
    phase4 = "phase4" # Post-conflict (control) roll

    def rollkeys(self):
        return [
            self.phase1,
            self.phase2,
            self.phase3,
            self.phase4
        ]


"""
Keys used for the Shaverdian et al. (2022) conflict experimental phases.
"""
class constkeys_conflict_shav_result_dict():
    __slots__ = ()
    phase1 = "phase1" # First roll
    phase2 = "phase2" # Second roll
    phase3 = "phase3" # Pre-conflict roll
    phase4 = "phase4" # Initial conflict
    phase5 = "phase5" # Control roll
    phase6 = "phase6" # Second conflict roll
    phase7 = "phase7" # Control
    phase8 = "phase8" # Control, need all eight to implement exlcusion criterion
    conflict_orders = "conflict_order" # The order in which conflicts were presented

    def rollkeys(self):
        return [
            self.phase1,
            self.phase2,
            self.phase3,
            self.phase4,
            self.phase5,
            self.phase6,
            self.phase7,
            self.phase8
        ]

"""
Keys used in specifying simulations in simulation_utilities.py
"""
class constkeys_simulation_param_dict():
    __slots__ = ()
    c2_offsets = "c2_offset"
    desired = "desired"
    k1 = "k1"
    k2 = "k2"
    w1 = "w1"
    w2 = "w2"
    s1 = "s1"
    s2 = "s2"
    timeout = "timeout"
    n_rolls = "n_rolls"
    max_turn = "max_turn"
    step_size = "step_size"
    arena_size = "arena_size"
    sm_seed = "sm_seed"
    sm_noise = "sm_noise"
    flatten_on_first = "flatten_on_first"
    reseed_generators = "reseed_generators"
    desired_angle_seed = "desired_angle_seed"
    placement_angle_seed = "placement_angle_seed"
    cue_one_noise_gen = "cue_one_noise_gen"
    cue_two_noise_gen = "cue_two_noise_gen"
    sm_noise_gen = "sm_noise_gen"
    placement_gen = "placement_gen"
    desired_direction_gen = "desired_direction_gen"
    n = "n"

"""
Keys for parameter dictionary used by the RingModel class in
extended_ring_model.py
"""
class constkeys_rm_param_dict():
    __slots__ = ()
    lr = "lr"
    r_threshold = "r_threshold"
    epg_threshold = "epg_threshold"
    n_r1 = "n_r1"
    n_r2 = "n_r2"
    n_r = "n_r"
    w_r_epg = "w_r_epg"
    w_epg_peg = "w_epg_peg"
    w_epg_pen = "w_epg_pen"
    w_epg_d7 = "w_epg_d7"
    w_d7_peg = "w_d7_peg"
    w_d7_pen = "w_d7_pen"
    w_d7_d7 = "w_d7_d7"
    w_peg_epg = "w_peg_epg"
    w_pen_epg = "w_pen_epg"
    w_sm_pen = "w_sm_pen"
    r_slope = "r_slope"
    r_bias = "r_bias"
    epg_slope = "epg_slope"
    epg_bias = "epg_bias"
    d7_slope = "d7_slope"
    d7_bias = "d7_bias"
    peg_slope = "peg_slope"
    peg_bias = "peg_bias"
    pen_slope = "pen_slope"
    pen_bias = "pen_bias"
    d_w1 = "d_w1"
    d_w2 = "d_w2"
    dynamic_r_inhibition = "dynamic_r_inhibition"
    r_inhibition = "r_inhibition"
    show_inputs = "show_inputs"
    verbose = "verbose"

"""
Keys for the decoding dictionary used by the RingModel class in
extended_ring_model.py (RingModel.decode(self))
"""
class constkeys_rm_decode_dict():
        __slots__ = ()
        r1 = "r1"
        r2 = "r2"
        epg = "epg"
        d7 = "d7"
        peg = "peg"
        pen = "pen"
        r1_epg = "r1_epg"
        r2_epg = "r2_epg"

"""
[Legacy]
Keys for the input dictionary used by the RingModel class in
extended_ring_model.py. This dictionary is updated with each
neuron class which receives input. The information can (to some
extent) be used to auto-tune the activation function but I
saw limited success with this.
"""
class constkeys_rm_input_dict():
        __slots__ = ()
        epg = "epg"
        d7 = "d7"
        pen = "pen"
        peg = "peg"

"""
Keys for the summary stats dictionary used by simulation_utilities.py
(summary_statistics()).
"""
class constkeys_summary_stats_dict():
    __slots__ = ()
    avg_distance = "avg_distance"
    avg_time = "avg_time"
    exit_vectors = "exit_vectors"
    mean_vectors =  "mean_vectors"
    head_directions = "head_directions"
    mean_heading_vectors = "mean_heading_vectors"
    exit_angles = "exit_angles"
    overall_mean = "overall_mean"
    overall_avg_dist = "overall_avg_dist"


"""
Keys for data structure returned by simulation_utilities.perform_runs()
"""
class constkeys_simdata_dict():
    __slots__ = ()
    agent = "agent"
    world = "world"
    snapshots = "snapshots"


"""
Conflict_weight.py runs four separate simulation routines which
each have the same structure but slightly different parameterisations.
The parameterisations are stored in a dictionary, the keys are
provided here.
"""
class constkeys_conflict_weight_params_dict():
    __slots__ = ()
    vary_strength = "vary_strength"
    vary_reliability = "vary_reliability"
    vary_both = "vary_both"
    vary_both_inv = "vary_both_inv"
    all_equal_k4 = "all_equal_k4"
    all_equal_k2 = "all_equal_k2"

"""
Simulation parameterisation keys for conflict_dacke.py.
"""
class constkeys_conflict_dacke_params_dict():
    __slots__ = ()
    high = "high"
    low = "low"
    mid = "mid"

#
# Instantiations - to be imported with the module
#

# Ring model keys
rmkeys = constkeys_rm_param_dict()
decodekeys = constkeys_rm_decode_dict()
inputkeys = constkeys_rm_input_dict()

# Simulation
simkeys = constkeys_simulation_param_dict()
statskeys = constkeys_summary_stats_dict()
simdatakeys = constkeys_simdata_dict()

# Precision experiment
precision_reskeys = constkeys_precision_experiment_result_dict()
modality_transfer_reskeys = constkeys_modality_transfer_experiment_result_dict()

# Conflict weighting experiment
cwparamkeys = constkeys_conflict_weight_params_dict()
conflict_weight_reskeys = constkeys_conflict_weight_result_dict()

# Dacke et al. (2019) conflict experiment
dackeparams = constkeys_conflict_dacke_params_dict()

# Shaverdian et al. (2022) conflict experiment
conflict_shav_reskeys = constkeys_conflict_shav_result_dict()

