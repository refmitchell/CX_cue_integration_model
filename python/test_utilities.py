"""
test_utilities.py

Miscellaneous utility functions. For the most part this module serves as
a custom circular statistics library.

References:
Shaverdian et al. (2022) - Weighted Cue Integration for Straight-line orientation
Wilkie (1983) - Miscellanea Rayleigh test for Randomness of Circular Data
Batschelet (1981) - Circular Statistics in Biology
Zar (2010) - Biostatistical Analysis
Berens (2009) - MATLAB Circular Statistics Toolbox
"""

import numpy as np
import matplotlib.pyplot as plt
from extended_ring_model import *
from dict_key_definitions import rmkeys, decodekeys
from scipy.stats import chi2, norm


def v_test(input_angles, predicted_mean):
    """
    V test for determining directedness of a dataset with prior knowledge
    about what the mean may be. See Batschelet (1981)

    Code adapted from Berens (2009). This implementation assumes no binning of
    data.

    :param input_angles: The angular sample data (radians)
    :param predicted_mean: The predicted mean direction (radians)
    :param circular_mean: (Optional) include a mean tuple if this has already
                          been computed.
    :return (p, v): The p-value and V statistic.
    """
    # Sanitise
    a = [x for x in input_angles if not np.isnan(x)]
    n = len(a)
    r, th = circmean(a) # imported from simulation_utilities.py
    R = n*r

    # Berens - gives different V statistic, but same significance
    # v = R * np.cos(th - predicted_mean)
    # u = v*np.sqrt(2/n)

    # Batschelet
    v = r * np.cos(th - predicted_mean)
    u = v*np.sqrt(2*n)
    p = 1 - norm.cdf(u)

    return (p, v)

def circmean(angles,
             scale=1,
             weights=None,
             include_kappa=False):
    """
    Compute the mean vector for an set of sample angles. Note that if you
    want to use this to plot the mean vector alongside some tracks,
    the mean vector will need to be re-scaled to match whatever size
    the arena was. This can be done using the scale parameter but is
    better done using the normal 0,1 r-value.

    Weights are permitted but you MUST check that none of your input angles
    are NaN values. If any of the input angles are NaN values then the function
    will fall back on an unweighted mean with a warning message.

    :param angles: The set of sample angles (radians).
    :param scale: Scaling factor for the mean vector (of use if you want to map
                  the mean vector onto a plot which works at arena scale).
    :param include_kappa: Approximate the concentration (kappa) of the von Mises
                          distribution from which this sample is most likely to
                          have been drawn. Warning: changes the form of the
                          output.
    :return: a tuple of the form (r, theta) the mean vector length and mean angle

    """
    angles = [x for x in angles if not np.isnan(x)] # Filter out NaN angles

    if len(angles) == 0:
        return (0,0)

    if weights == None:
        weights = [ 1 for x in angles ]

    if len(weights) != len(angles):
        print("Warning: Length of weight array and angle array size mismatch in")
        print("         mean vector computation. Either this is an error or one") 
        print("         of your input angles is NaN. Defaulting to unweighted")
        print("         mean.")
        weights = [ 1 for x in angles ]


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

def k(p, n):
    """
    Standard method from Wilkie (1983)
    (from Shaverdian et al. (2022) repository).
    :param p: significance
    :param n: sample size
    """
    return -np.log(p) - ( (2*np.log(p) + np.log(p)**2) / (4*n) )

def rayleigh_Z(n, r):
    """
    Compute rayligh Z statistic. From Shaverdian et al. (2022),
    drawn from Batschelet (1981).

    :param n: sample size
    :param r: r-value (mean vector length)
    """
    return n * (r**2)

def rayleigh_crit(p, n, r):
    """
    Given significance, sample size, and r-value; determine
    if r-val is statistically significant.

    From Shaverdian et al. (2022), drawn from Batschelet (1981)
    :param p: desired significance
    :param n: sample size
    :param r: r-value (mean vector length)
    """
    Z = rayleigh_Z(n, r)
    K = k(p, n)

    if Z > K:
        return True

    return False

def confidence_interval(r, significance, n):
    """
    Method for computing the confidence interval on the mean angle. This
    method is from Biostatistical Analysis by Jerrold H. Zar, Fifth Edition
    (ISBN 13: 978-1-292-02404-2), Equations 24 and 25 given on p. 658

    Dependencies: scipy, numpy

    :param r: The mean vector length of the sample (usually denoted 'r').
    :param significance: The significance to which you want to compute the
                         confidence interval. E.g. if you wanted the 95% CI,
                         you would use 0.05, 99% -> 0.01.
    :param n: The sample size.
    :return d: The confidence interval. Note that this function does not return
               the upper and lower limits; these can be easily computed as
               mean angle + d and mean angle - d respectively.
    """
    R = n*r
    chival = chi2.isf(significance, 1)
    if  r >= 0.9:
        d = np.arccos(
            np.sqrt(n**2 - (n**2 - R**2) * np.exp(chival/n)) / R
        )
        return d
    # print("n: {}".format(n))
    # print("X: {}".format(chival))
    # print("R: {}".format(R))
    # print("2*R**2: {}".format(2*R**2))
    # print("n*X: {}".format(n*chival))
    root_arg = (2*n*(2*(R**2) - n*chival)) / (4*n - chival)
    # print(root_arg)
    d = np.arccos(
        np.sqrt(root_arg) / R
    )

    return d

def angular_deviation(r, s0=True):
    """
    Method for computing the angular deviation (circular std. dev.) of a sample.
    Method taken from Batschelet (1981).

    :param r: The mean vector length of the sample
    :param s0: Angular deviation has a couple of different definitions. Usually
               these are denoted s and s0. s0 seems to be preferred (Wikipedia,
               Batschelet, and Zar all use s0 primarily) so by default s0 is
               True. If False, the method will return s instead.
    :return: The angular deviation (s or s0 depending on param s0)
    """

    # Note: Zar explicitly labels s as 'angular deviation' and s0 as 'circular
    # standard deviation'. Batschelet doesn't make the same distinction and they
    # both seem to cite the same work by Mardia (1972).
    if s0:
        # Batschelet (1981) Eq. 2.3.4 (same expression [for degree measurements]
        # is given by Zar in Ch. 26 of Biostatistical Analysis).
        return np.sqrt(-2*np.log(r))

    # Batschelet (1981) Eq. (2.3.2) (again given by Zar [for degrees] in Ch. 26
    # of Biostatistical Analysis).
    return np.sqrt(2*(1 - r))

def kappa(r):
    """
    Kappa estimation from mean vector length given by Mardia and Jupp (2009),
    used by Shaverdian et al. (2022) to estimate cue noise distributions from
    population behaviour under single cue conditions. This version has been
    modified to work on numpy ndarrays rather than just scalars.
    """

    # Make sure r is an array
    r = np.array(r)
    kappas = np.zeros(r.shape)

    # Use 'where' to find relevant indices for each condition,
    # insert kappa if the condition is satisfied, zero otherwise.
    kappas += np.where(r < 0.53, 2*r + r**3 + (5/6)*(r**5) , 0)
    kappas += np.where(r >= 0.85, 1 / (2*(1 - r) - (1 - r)**2 - (1 - r)**3), 0)
    kappas += np.where((r >= 0.53) & (r < 0.85), -0.4 + 1.39*r + (0.43/(1 - r)), 0)

    # [()] returns a scalar if kappas is 0-dimensional (i.e. a scalar) and
    # returns kappas as an ordinary ndarray otherwise. Solves problems
    # further up the chain where kappas are expected to be of scalar type.
    return kappas[()]

def circ_scatter(angles, radial_base=1, radial_interval=0.1):
    """
    Given a series of angles, return a list of associated radii, such
    that each duplicated angle gets an increased radius. Specifically
    for polar scatter plots.

    :param angles: The angular data in question
    :param radial_base: The base radius of all datapoints.
    :param radial_interval: The radial distance added with each duplicate.
    """
    unique, counts = np.unique(angles, return_counts=True)

    radii = []
    angle_out = []
    for idx in range(len(unique)):
        radius = radial_base
        for c in range(counts[idx]):
            # For each instance of this unique value, increment radius
            # and add angle back to the output list.
            radii.append(radius)
            angle_out.append(unique[idx])
            radius += radial_interval

    # angle_out = list of angles, should be equivalent to angles input
    # radii = list of corresponding radii
    return radii, angle_out

def angular_difference(a1, a2, signed=False):
    """
    Compute the difference between two angles. If signed is true then.
    the order of the arguments matters.

    :param a1: Angle 1
    :param a2: Angle 2
    :signed: Compute the signed angle rather than the inner angle
    """

    v1 = [np.cos(a1), np.sin(a1)]
    v2 = [np.cos(a2), np.sin(a2)]

    dot = v1[0]*v2[0] + v1[1]*v2[1]
    m1 = np.sqrt(np.sum([x**2 for x in v1]))
    m2 = np.sqrt(np.sum([x**2 for x in v2]))

    if signed:
        v1 = [np.cos(a1), np.sin(a1)]
        v2 = [np.cos(a2), np.sin(a2)]
        det = v1[0]*v2[1] - v1[1]*v2[0]
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        ret = np.arctan2(det,dot)
    else:
        ret = np.arccos(dot / (m1*m2))

    return ret

def dynamic_parameterise(x=[ 0.49999982,
                             0.49999982,
                             0.49999982,
                             0.50000054,
                             -0.01,
                             -0.01000071,
                              0.49999982,
                             0.49999982,
                             0.49999982,
                             0.50000054,
                             3.99999857,
                             4.00000215,
                             4.00000215,
                             4.00000215,
                             4.00000215,
                             4.00000215,
                             4.00000215,
                             4.00000215,
                             4.00000215,
                             3.99999857]
):
    """
    [Legacy] -Take a vector of dimension 10 and return a ring model using the
    same parameterisation conventions used in optimize_dynamics. Included here
    for easy import into other modules to ease immediate testing.

    Default is current optimum.
    :param x:
    :return rm:
    """
    params = dict()

    # Weights
    params[rmkeys.w_r_epg] = x[0]
    params[rmkeys.w_epg_peg] = x[1]
    params[rmkeys.w_epg_pen] = x[2]
    params[rmkeys.w_epg_d7] = x[3]
    params[rmkeys.w_d7_peg] = x[4]
    params[rmkeys.w_d7_pen] = x[5]
    params[rmkeys.w_d7_d7] = x[6]
    params[rmkeys.w_peg_epg] = x[7]
    params[rmkeys.w_pen_epg] = x[8]
    params[rmkeys.w_sm_pen] = x[9]

    # Activation functions
    params[rmkeys.r_slope] = x[10]
    params[rmkeys.r_bias] = x[11]
    params[rmkeys.epg_slope] = x[12]
    params[rmkeys.epg_bias] = x[13]
    params[rmkeys.d7_slope] = x[14]
    params[rmkeys.d7_bias] = x[15]
    params[rmkeys.peg_slope] = x[16]
    params[rmkeys.peg_bias] = x[17]
    params[rmkeys.pen_slope] = x[18]
    params[rmkeys.pen_bias] = x[19]

    return RingModel(params)

def decode_model(rm):
    """
    [Legacy] - Decoding was built directly into the RingModel class.

    Helper method to decode the current angle/magnitude stored in the E-PG layer
    of a given RingModel.

    :param rm: an extended_ring_model.RingModel
    """
    polar_epg_vectors = zip(rm.epg_prefs, rm.epg_rates)
    cart_epg_vectors = [ (r*np.cos(t), r*np.sin(t)) for (t,r) in polar_epg_vectors]
    xs = [ x for (x,_) in cart_epg_vectors ]
    ys = [ y for (_,y) in cart_epg_vectors ]

    x_avg = sum(xs) / len(ys)
    y_avg = sum(ys) / len(ys)

    r_avg = np.sqrt(x_avg**2 + y_avg**2)
    t_avg = np.arctan2(y_avg, x_avg)
    t_avg = t_avg[0] # need to find why this returns a list...
    return t_avg, r_avg

def mmcs(a1, a2, w1, w2):
    """
    Murray and Morgenstern polar weighted sum

    :param a1: angle one
    :param a2: angle two
    :param w1: weight one
    :param w2: weight two
    """
    # Eq. 16 from Murray and Morgenstern, 2010
    l = a2 + np.arctan2(
        np.sin(a1 - a2),
        (w2 / w1) + np.cos(a1 - a2))
    k = np.sqrt(w2**2 + w1**2 + 2*w2*w1*np.cos(a1 - a2))
    return l, k

def plasticity_tuning(animation=False):
    """
    [Legacy] - Helper function used to tune learning parameters.
    """
    rm = RingModel(static_init=True)
    duration = 15 # Simulation duration - degrees

    # Starting angles
    c1 = 0
    c2 = 179

    r1_frames = np.zeros((duration,8,8))
    r2_frames = np.zeros((duration,8,8))

    # Timeseries
    r1 = np.zeros((8,duration))
    r2 = np.zeros((8,duration))
    epg = np.zeros((8,duration))
    d7 = np.zeros((8, duration))
    pen = np.zeros((16,duration))
    peg = np.zeros((16,duration))

    """
    Stabilisation routine: allows time for bump to stabilise after
    initialisation before performing learning. This is relevant for
    large conflicts
    """
    for t in range(500):
        rm.update_state(c1,
                        c2,
                        sm=24,
                        plasticity=False
        )
    """
    Learning routine
    """
    total_rotation=0

    # 'zero' the adjacency matrices for learning
    rm.w_r1_epg = np.zeros((8,8)) + 0.01
    rm.w_r2_epg = np.zeros((8,8)) + 0.01

    c1=0
    c2=179
    for t in range(duration):
        angle_update = 24 # 6deg per timestep

        total_rotation+=angle_update

        c1+=angle_update
        c2+=angle_update

        rm.update_state(c1,
                        c2,
                        sm=angle_update,
                        plasticity=True
        )

        r1_frames[t] = rm.w_r1_epg
        r2_frames[t] = rm.w_r2_epg

        # Logging
        r1[:,t] = rm.r1_rates.reshape(8)
        r2[:,t] = rm.r2_rates.reshape(8)
        epg[:,t] = rm.epg_rates.reshape(8)
        print(rm.epg_rates.reshape(8))
        d7[:,t] = rm.d7_rates.reshape(8)
        pen[:,t] = rm.pen_rates.reshape(16)
        peg[:,t] = rm.peg_rates.reshape(16)

    if animation == False:
        fig = plt.figure()
        ax = plt.subplot(611)
        ax.pcolormesh(r1, vmin=0, vmax=1)
        ax =plt.subplot(612)
        ax.pcolormesh(r2, vmin=0, vmax=1)
        ax2 = plt.subplot(613)
        ax2.pcolormesh(epg, vmin=0, vmax=1)
        ax3 = plt.subplot(614)
        ax3.pcolormesh(d7, vmin=0, vmax=1)
        ax4 = plt.subplot(615)
        ax4.pcolormesh(pen, vmin=0, vmax=1)
        ax5 = plt.subplot(616)
        ax5.pcolormesh(peg, vmin=0, vmax=1)
        # Display final weight status
        fig = plt.figure()
        w1_ax = plt.subplot(121)
        w2_ax = plt.subplot(122)

        # Data
        w1_ax.pcolormesh(rm.w_r1_epg, vmin=0, vmax=1, cmap='hot')
        w2_ax.pcolormesh(rm.w_r2_epg, vmin=0, vmax=1, cmap='hot')

        # Appearance
        # W1
        w1_ax.set_ylabel("E-PG index")
        w1_ax.set_xlabel("R-index")
        w1_ax.set_ylim([8,0])
        w1_ax.set_xlim([0,8])
        w1_ax.set_title("R1 -> E-PG adjacency")

        # W2
        w2_ax.set_ylabel("E-PG index")
        w2_ax.set_xlabel("R-index")
        w2_ax.set_ylim([8,0])
        w2_ax.set_xlim([0,8])
        w2_ax.set_title("R2 -> E-PG adjacency")
        plt.show()

    else:
        #
        # Animation plotting
        #
        fig = plt.figure()
        w1_ax = plt.subplot(121)
        w2_ax = plt.subplot(122)
        fig.tight_layout()

        for t in range(len(r1_frames)):
            r1_epg = r1_frames[t]
            r2_epg = r2_frames[t]
            filename = "{:05d}.png".format(t)

            w1_ax.clear()
            w2_ax.clear()

            # Data
            w1_ax.pcolormesh(r1_epg, vmin=0, vmax=1, cmap='hot')
            w2_ax.pcolormesh(r2_epg, vmin=0, vmax=1, cmap='hot')

            # Appearance
            # W1
            w1_ax.set_ylabel("E-PG index")
            w1_ax.set_xlabel("R-index")
            w1_ax.set_ylim([8,0])
            w1_ax.set_xlim([0,8])
            w1_ax.set_title("R1 -> E-PG adjacency")

            # W2
            w2_ax.set_ylabel("E-PG index")
            w2_ax.set_xlabel("R-index")
            w2_ax.set_ylim([8,0])
            w2_ax.set_xlim([0,8])
            w2_ax.set_title("R2 -> E-PG adjacency")

            plt.savefig("plots/class_weight_frames/{}".format(filename))

            print("{}, done.".format(filename))

        return rm

def simple_timeseries():
    """
    [Legacy] - replaced by the timeseries module
    """
    rm = RingModel(static_init=True)
    duration = 1000

    # Starting angles
    c1 = 0
    c2 = 179

    r1_frames = np.zeros((duration,8,8))
    r2_frames = np.zeros((duration,8,8))

    # Timeseries
    r1 = np.zeros((8,duration))
    r2 = np.zeros((8,duration))
    epg = np.zeros((8,duration))
    d7 = np.zeros((8, duration))
    pen = np.zeros((16,duration))
    peg = np.zeros((16,duration))

    for t in range(duration):
        angle_update = np.random.normal(loc=0, scale=np.sqrt(100)) # 6deg per timestep

        c1+=angle_update
        c2+=angle_update

        p = False
        if t >400 and t < 600:
            p = True

        rm.update_state(c1,
                        c2,
                        sm=angle_update,
                        disable_rs=p
        )

        r1_frames[t] = rm.w_r1_epg
        r2_frames[t] = rm.w_r2_epg

        # Logging
        r1[:,t] = rm.r1_rates.reshape(8)
        r2[:,t] = rm.r2_rates.reshape(8)
        epg[:,t] = rm.epg_rates.reshape(8)
        d7[:,t] = rm.d7_rates.reshape(8)
        pen[:,t] = rm.pen_rates.reshape(16)
        peg[:,t] = rm.peg_rates.reshape(16)

    fig = plt.figure()
    ax = plt.subplot(611)
    ax.pcolormesh(r1, vmin=0, vmax=1)
    ax =plt.subplot(612)
    ax.pcolormesh(r2, vmin=0, vmax=1)
    ax2 = plt.subplot(613)
    ax2.pcolormesh(epg, vmin=0, vmax=1)
    ax3 = plt.subplot(614)
    ax3.pcolormesh(d7, vmin=0, vmax=1)
    ax4 = plt.subplot(615)
    ax4.pcolormesh(pen, vmin=0, vmax=1)
    ax5 = plt.subplot(616)
    ax5.pcolormesh(peg, vmin=0, vmax=1)

    # fig = plt.figure()
    # w1_ax = plt.subplot(121)
    # w2_ax = plt.subplot(122)

    # # Data
    # w1_ax.pcolormesh(rm.w_r1_epg, vmin=0, vmax=1, cmap='hot')
    # w2_ax.pcolormesh(rm.w_r2_epg, vmin=0, vmax=1, cmap='hot')

    # # Appearance
    # # W1
    # w1_ax.set_ylabel("E-PG index")
    # w1_ax.set_xlabel("R-index")
    # w1_ax.set_ylim([8,0])
    # w1_ax.set_xlim([0,8])
    # w1_ax.set_title("R1 -> E-PG adjacency")

    # # W2
    # w2_ax.set_ylabel("E-PG index")
    # w2_ax.set_xlabel("R-index")
    # w2_ax.set_ylim([8,0])
    # w2_ax.set_xlim([0,8])
    # w2_ax.set_title("R2 -> E-PG adjacency")
    plt.show()
