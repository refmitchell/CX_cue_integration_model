"""
estimate_weights.py

Provides crude weight estimation for cue configurations from Shaverdian
et al. (2022). We assume cue inputs can be described by vectors
pointing towards the true position of the cue. This script then attempts
to find a combination of magnitudes in order to minimise the difference
between the sum of the cue vectors and the sample means.

[Legacy] - This was used as an initial exploration for cue conflict, we
used the conflict data from Shaverdian et al. (2022) to try and
determine cue weights for different conditions. This ended up not
being particularly useful. 

"""


import json
import pandas as pd
import numpy as np
from test_utilities import confidence_interval, angular_deviation, circmean

import matplotlib.pyplot as plt

if __name__ == "__main__":
    changes = pd.read_csv("shav2022_data/changes_full.csv")
    speeds = [1.25, 2.50]
    elevations = [45, 60, 75, 86]
    conflicts = [60,120]
    results = dict()

    for e in elevations:
        for s in speeds:
            if e == 45 and s == 1.25:
                continue
            exp = "{}-{:.2f}".format(e,s)


            # Initial minimum
            weights = np.linspace(0,1,100) # 'strengths' would be more correct

            # Fix sun strength using projection
            w1 = np.cos(np.radians(e))

            v1 = np.array([np.cos(0), np.sin(0)])
            v2 = np.array([0,0])
            res = v1 + v2
            min_avg_diff = -1
            min_ws = [1,0]

            for w2 in weights:
                # For each possible w2, set its weight relative to fixed
                # w1 then compute the error.
                w1 = w1 / (w1 + w2)
                w2 = 1 - w1

                diff = 0
                for c in conflicts:
                    data = np.array(changes["{:.02f}-{}-{:03d}".format(s,e,c)])
                    data = data[~np.isnan(data)]

                    # Verified against Shaverdian et al. (2022) reported mean and std. dev.
                    r,t =  circmean(np.radians(data))
                    goal = np.array([r*np.cos(t), r*np.sin(t)])

                    v1 = np.array([w1*np.cos(0), w1*np.sin(0)])
                    v2 = np.array([w2*np.cos(np.radians(c)), w2*np.cos(np.radians(c))])
                    res = v1 + v2
                    diff += np.sqrt((goal[0] - res[0])**2 + (goal[1] - res[1])**2)

                    # Cartesian cue vectors
                    avg_diff = diff/len(conflicts)

                if min_avg_diff < 0:
                    min_avg_diff = avg_diff
                    min_ws = [w1, w2]

                if avg_diff < min_avg_diff:
                    min_avg_diff = avg_diff
                    min_ws = [w1, w2]

                results[exp] = dict()
                results[exp]["weights"] = (min_ws[0],min_ws[1])
                results[exp]["minimum_average_error"] = min_avg_diff

    with open("data_out/weights_from_shav.json", "w") as f:
        json.dump(results, f)

