"""
plot_simple_conflict.py

[Legacy] - This was the precursor to plot_conflict_weight.py
"""

import json
import numpy as np
import itertools
import matplotlib.pyplot as plt

import test_utilities as testutils
import dict_key_definitions as keydefs
from dict_key_definitions import statskeys, simkeys
from dict_key_definitions import modality_transfer_reskeys as reskeys

def model_prediction(k1,k2,cue_one,cue_two):
    w1 = k1 / (k1 + k2)
    w2 = k2 / (k1 + k2)
    c1 = np.radians(cue_one)
    c2 = np.radians(cue_two)

    xs = [w1*np.cos(c1), w2*np.cos(c2)]
    ys = [w1*np.sin(c1), w2*np.sin(c2)]
    x = sum(xs)
    y = sum(ys)

    r = np.sqrt(x**2 + y**2)
    t = np.arctan2(y,x)

    return (r,t)




if __name__ == "__main__":
    # Load stats data from file
    with open("data_out/simple_conflict_stats_vb.json") as f:
        stats = json.load(f)

    # Load simulation metadata
    with open("data_out/simple_conflict_meta_vb.json") as f:
        meta = json.load(f)

    # Exit angles, should be one entry for each agent (n = 40).
    # Rolls 1 and 2 done in roll1, 3 in roll2,
    # 4 in roll3 and the final control in roll4

    # roll2 is correct.
    roll_three_exits = stats[reskeys.roll2][statskeys.exit_angles]
    roll_three_exits = list(itertools.chain(*roll_three_exits))
    roll_three_exits = np.radians(
        np.around(np.degrees(roll_three_exits)/5, decimals=0)*5
    )

    # Again, roll3 is correct
    roll_four_exits = stats[reskeys.roll3][statskeys.exit_angles]
    roll_four_exits = list(itertools.chain(*roll_four_exits))
    roll_four_exits = np.radians(
        np.around(np.degrees(roll_four_exits)/5, decimals=0)*5
    )

    changes = (roll_four_exits - roll_three_exits)
    mean = testutils.circmean(changes)
    unique, counts = np.unique(changes, return_counts=True)

    print(mean[0])

    # Work out rough predicted mean
    c1 = 0
    c2 = 120
    r,t = model_prediction(4, 4, c1, c2)
    r1,t1 = model_prediction(1, 4, c1, c2)

    plt.subplot(111, projection="polar")
    plt.title("Different strengths and different reliabilities")
    plt.scatter(changes, np.ones(len(changes)), alpha=0.5)
    plt.gca().set_theta_direction(-1)
    plt.gca().set_theta_zero_location("N")
    plt.arrow(0, 0, mean[1],mean[0], color='k', label="Population mean")
    plt.arrow(0,0,t,r, color='red', label="Equal reliabilities")
    plt.arrow(0,0,t1,r1,color='orange', label="Actual reliabilities")
    plt.ylim([0,1.1])
#    plt.legend()

    plt.savefig("conflict_vary_both.pdf", bbox_inches="tight")
    plt.show()
