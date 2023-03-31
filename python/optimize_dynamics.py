"""
[Legacy]
optimize_dynamics.py

Routine which attempted to determine network parameters
using basic desirable network characteristics as objectives.
"""

import smtplib
import numpy as np
import matplotlib.pyplot as plt
from test_utilities import mmcs, dynamic_parameterise
from extended_ring_model import *
from dict_key_definitions import rmkeys, decodekeys

from scipy import optimize

it = 0
def progress(xk, convergence=None):
    global it

    rm = dynamic_parameterise(x=xk)

    log = "\n === Optimisation progress ===\n"
    log += "It: {}, xk: {}\n".format(it,xk)
    log += "Conflict err: {}\n".format(conflict_comparison(rm))
    log += "Absence err : {}\n".format(cue_absence(rm))
    log += "Dominance err: {}\n".format(r_dominance(rm))
    log += "Convergence: {}\n".format(convergence)
    log += "\n === Message end ===\n"

    sendmail_on_callback(text=log)

    print(log)

    it+=1
    return False

def angular_error(a1, a2, degrees=False):
    """
    Compute the absolute angular error between two angles.
    :param a1: Angle 1
    :param a2: Angle 2
    :param degrees: arguments and result in degrees
    :return: Unsigned difference between two angles
    """
    if degrees:
        a1 = np.radians(a1)
        a2 = np.radians(a2)

    a = [np.cos(a1), np.sin(a1)]
    b = [np.cos(a2), np.sin(a2)]
    a_dot_b = sum([ x*y for (x,y) in zip(a,b) ])
    mag_a = np.sqrt(np.sum([x**2 for x in a]))
    mag_b = np.sqrt(np.sum([x**2 for x in b]))
    cos_theta = a_dot_b/(mag_a*mag_b)
    cos_theta = np.clip(cos_theta, -1, 1)
    print(cos_theta)
    res = np.arccos(cos_theta)

    if degrees:
       res =  np.degrees(res)

    return res

def conflict_comparison(rm, n=500, reset=False):
    """
    Objective 1: Vector sum conflict behaviour
    Perform the standard conflict comparison paradigm over different weights.
    :param rm: the RingModel to be compared to the vector sum.
    :param reset: reset rates between each conflict test
    :return: error (E-PG angle vs vector sum [mmcs])
    """
    rm.initialise()
    weights = np.arange(0.1, 1, 0.1)
    conflicts = np.linspace(0,180,n)
    total_error = 0


    # fig = plt.figure()
    # mmcs_ax = plt.subplot(121)
    # ring_ax = plt.subplot(122)

    for w1 in weights:
        if reset: rm.reset_rates()
        w2 = 1 - w1
        errors = []
        ring_out = []
        mmcs_out = []
        c1 = 0
        for c2 in conflicts:
            rm.update_state(c1,
                            c2,
                            sm=0,
                            w1=w1,
                            w2=w2,
                            plasticity=False)
            r_t, _ = rm.decode()[decodekeys.epg]
            vs_t, _ = mmcs(np.radians(c1), np.radians(c2), w1, w2)
            ring_out.append(r_t)
            mmcs_out.append(vs_t)
            errors.append(angular_error(r_t, vs_t))

        # mmcs_ax.plot(mmcs_out)
        # ring_ax.plot(ring_out)
        print(sum(errors))
        total_error += sum(errors)

#    plt.show()

    # If we're not returning a plot, return here
    return total_error


def cue_absence(rm, scale=20, loc=0, duration=1000, treatment=400):
    """
    Objective 2: In the absence of external input, E-PGs should track
    with real position (sm integration).
    :param rm: the RingModel to be evaluated
    :return: error (E-PG vs real)
    """
    c1 = 0
    c2 = 0
    sm = 0
    treatment_start = (duration/2) - (treatment/2)
    treatment_end = (duration/2) + (treatment/2)
    rm.initialise()
    total_error = 0
    for t in range(duration):
        change = np.random.normal(loc=loc, scale=scale)
        c1 += change
        c2 += change
        w1 = 0.5
        w2 = 0.5
        if t >= treatment_start and t <= treatment_end:
            w1 = 0
            w2 = 0

        rm.update_state(c1, c2, w1=w1, w2=w2, sm=change)
        total_error += angular_error(rm.decode()[decodekeys.epg][0], np.radians(c1))

    return total_error


def r_dominance(rm, scale=20, loc=0, duration=1000, treatment=400):
    """
    Objective 3: If an R signal is present, they should primarily drive the
    E-PGs. E-PG and R s
    :param rm: the RingModel to be evaluated
    :return: error (E-PG vs R)
    """
    c1 = 0
    c2 = 0
    sm = 0
    treatment_start = (duration/2) - (treatment/2)
    treatment_end = (duration/2) + (treatment/2)
    rm.initialise()
    total_error = 0
    for t in range(duration):
        change = np.random.normal(loc=loc, scale=scale)
        c1 += change
        c2 += change
        w1 = 0.5
        w2 = 0.5
        if t >= treatment_start and t <= treatment_end:
            w1 = 0
            w2 = 0

        rm.update_state(c1, c2, w1=w1, w2=w2, sm=change)
        total_error += angular_error(rm.decode()[decodekeys.epg][0],
                                     rm.decode()[decodekeys.r1][0])

    return total_error

def objective(x):
    """
    Formulation for optimisation
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

    err = 0

    rm = RingModel(params)
    err = conflict_comparison(rm)

    rm = RingModel(params) # Fine to re-init each time
    err += cue_absence(rm)

    rm = RingModel(params) # Fine to re-init each time
    err += r_dominance(rm)

    return err


def main():
    w_lo = 0.01
    w_hi = 2
    weight_bounds = [
        (w_lo,w_hi), # R -> E-PG
        (w_lo,w_hi), # E-PG -> P-EG
        (w_lo,w_hi), # E-PG -> P-EN
        (w_lo,w_hi), # E-PG -> D7
        (-w_hi,-w_lo), # D7 -> P-EG (inhibitory)
        (-w_hi,-w_lo), # D7 -> P-EN (inhibitory)
        (w_lo,w_hi), # D7 -> D7 (inhibitory *)
        (w_lo,w_hi), # P-EG -> E-PG
        (w_lo,w_hi), # P-EN -> E-PG
        (w_lo,w_hi), # SM -> P-EN
    ]

    a_hi = 10
    a_lo = 0
    activation_bounds = [
        (a_lo, a_hi),
        (a_lo, a_hi),
        (a_lo, a_hi),
        (a_lo, a_hi),
        (a_lo, a_hi),
        (a_lo, a_hi),
        (a_lo, a_hi),
        (a_lo, a_hi),
        (a_lo, a_hi),
        (a_lo, a_hi)
    ]

    # Guess...
    bounds = weight_bounds + activation_bounds

    # Okay...
    x = optimize.differential_evolution(objective,
                                        bounds,
                                        callback=progress
    )

    # xk = [ 0.84678987,
    #        1.2946288,
    #        0.62027757,
    #        0.27777678,
    #        -0.50260314,
    #        -1.26128951,
    #        0.46861528,
    #        0.40384922,
    #        0.8133938,
    #        1.14279511,
    #        1.87374216,
    #        3.76774031,
    #        0.63750475,
    #        6.82320806,
    #        5.31131662,
    #        6.9135346,
    #        7.20759686,
    #        8.00639888,
    #        7.44978778,
    #        6.03181121 ]
    # rm = dynamic_parameterise(x=xk)
    # err = conflict_comparison(rm)

    # x = optimize.minimize(objective,
    #                       start,
    #                       bounds=bounds,
    #                       callback=progress)

    out = "COMPLETE\n" + str(x)

    sendmail_on_callback(text=out)

def sendmail_on_callback(text=""):
    """
    Send email to self for progress report. Note this may not work
    on other systems.
    :param text: Text to be included in the email.
    """
    sender = "r.mitchell@ed.ac.uk"
    receiver = ["r.mitchell@ed.ac.uk"]

    headers = "From: {}\nTo:{}\nSubject: Optimisation progress\n".format(sender, receiver[0])
    message = """{}Progress report:\n{}""".format(headers, text)

    server = smtplib.SMTP('localhost')
    server.sendmail(sender, receiver, message)
    server.quit()


if __name__ == "__main__":
    main()
