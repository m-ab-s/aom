#!/usr/bin/env python
## Copyright (c) 2019, Alliance for Open Media. All rights reserved
##
## This source code is subject to the terms of the BSD 2 Clause License and
## the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
## was not distributed with this source code in the LICENSE file, you can
## obtain it at www.aomedia.org/license/software. If the Alliance for Open
## Media Patent License 1.0 was not distributed with this source code in the
## PATENTS file, you can obtain it at www.aomedia.org/license/patent.
##
__author__ = "maggie.sun@intel.com, ryan.lei@intel.com"

import numpy
import math
import scipy.interpolate
import logging
from Config import LoggerName
from operator import itemgetter

subloggername = "CalcBDRate"
loggername = LoggerName + '.' + '%s' % subloggername
logger = logging.getLogger(loggername)


def Interpolate(qp, logbr, qty):
    '''
    generate interpolated RD sampling point based on QP values.
    using bilinear interpolation. bit rate are in the log domain
    incoming logbr and qty should be in the increasing order
    incoming qp could be in any order
    '''
    int_qp = []; int_logbr = []; int_qty = []
    for i in range(0, len(qp)):
        int_qp.append(qp[i])
        int_logbr.append(logbr[i])
        int_qty.append(qty[i])
        #handle duplicated qp
        if (i<len(qp)-1) and (qp[i] == qp[i+1]):
            continue
        #interpolate logbr and quality point between qp[i] and qp[i+1]
        #qp order is non-determinstics, because it could come from different resolution
        if (i<len(qp)-1) and ((qp[i] != qp[i + 1] + 1) or (qp[i] != qp[i + 1] - 1)):
            qlist = []
            if (qp[i+1] > qp[i]):
                qlist = range(qp[i] + 1, qp[i+1], 1)
            else:
                qlist = range(qp[i] - 1, qp[i+1], -1)
            for q in qlist:
                # bitrate(qp_target) = (bitrate(qp0) - bitrate(qp1)) / (qp0 - qp1) * (qp_target - qp0) + bitrate(qp0)
                # quality(qp_target) = (quality(qp0) - quality(qp1)) / (qp0 - qp1) * (qp_target - qp0) + quality(qp0)
                # result int_logbr and int_qty will always be in non-decreasing order.
                # but order of int_qp is similar as input qp
                br = (logbr[i] - logbr[i + 1]) / (qp[i] - qp[i + 1]) * (q - qp[i]) + logbr[i]
                qt = (qty[i] - qty[i + 1]) / (qp[i] - qp[i + 1]) * (q - qp[i]) + qty[i]
                int_qp.append(q)
                int_logbr.append(br)
                int_qty.append(qt)
    '''
    print("before interpolation:")
    for i in range(len(qp)):
        print("%d,  %f, %f"%(qp[i], logbr[i], qty[i]))
    print("after interpolation:")
    for i in range(len(int_qp)):
        print("%d,  %f, %f"%(int_qp[i], int_logbr[i], int_qty[i]))
    '''
    return int_qp, int_logbr, int_qty

# BJONTEGAARD    Bjontegaard metric
# Calculation is adapted from Google implementation
# PCHIP method - Piecewise Cubic Hermite Interpolating Polynomial interpolation
def BD_RATE(br1, qtyMtrc1, br2, qtyMtrc2, qp1=None, qp2=None, EnablePreInterpolation=False):
    brqtypairs1 = []; brqtypairs2=[]
    if (EnablePreInterpolation):
        assert(qp1 is not None and qp2 is not None)
        brqtypairs1 = [(br1[i], qtyMtrc1[i], qp1[i]) for i in range(min(len(qp1), len(qtyMtrc1), len(br1)))]
        brqtypairs2 = [(br2[i], qtyMtrc2[i], qp2[i]) for i in range(min(len(qp2), len(qtyMtrc2), len(br2)))]
    else:
        brqtypairs1 = [(br1[i], qtyMtrc1[i]) for i in range(min(len(qtyMtrc1), len(br1)))]
        brqtypairs2 = [(br2[i], qtyMtrc2[i]) for i in range(min(len(qtyMtrc2), len(br2)))]

    # sort the pair based on quality metric values in increasing order
    # if quality metric values are the same, then sort the bit rate in increasing order
    brqtypairs1.sort(key = itemgetter(1, 0))
    brqtypairs2.sort(key = itemgetter(1, 0))

    logbr1 = [math.log(x[0]) for x in brqtypairs1]
    qmetrics1 = [100.0 if x[1] == float('inf') else x[1] for x in brqtypairs1]
    logbr2 = [math.log(x[0]) for x in brqtypairs2]
    qmetrics2 = [100.0 if x[1] == float('inf') else x[1] for x in brqtypairs2]
    if (EnablePreInterpolation):
        qp1 = [x[2] for x in brqtypairs1]
        qp2 = [x[2] for x in brqtypairs2]

    if not brqtypairs1 or not brqtypairs2:
        logger.info("one of input lists is empty!")
        return 0.0

    # perform an bi-linear interpolation based on QP
    if EnablePreInterpolation:
        qp1, logbr1, qmetrics1 = Interpolate(qp1, logbr1, qmetrics1)
        qp2, logbr2, qmetrics2 = Interpolate(qp2, logbr2, qmetrics2)

    # remove duplicated quality metric value, the RD point with higher bit rate is removed
    dup_idx = [i for i in range(1, len(qmetrics1)) if qmetrics1[i - 1] == qmetrics1[i]]
    for idx in sorted(dup_idx, reverse=True):
        del qmetrics1[idx]
        del logbr1[idx]
    dup_idx = [i for i in range(1, len(qmetrics2)) if qmetrics2[i - 1] == qmetrics2[i]]
    for idx in sorted(dup_idx, reverse=True):
        del qmetrics2[idx]
        del logbr2[idx]

    # find max and min of quality metrics
    min_int = max(min(qmetrics1), min(qmetrics2))
    max_int = min(max(qmetrics1), max(qmetrics2))
    if min_int >= max_int:
        logger.info("no overlap from input 2 lists of quality metrics!")
        return 0.0

    # generate samples between max and min of quality metrics
    lin = numpy.linspace(min_int, max_int, num=100, retstep=True)
    interval = lin[1]
    samples = lin[0]

    # interpolation
    v1 = scipy.interpolate.pchip_interpolate(qmetrics1, logbr1, samples)
    v2 = scipy.interpolate.pchip_interpolate(qmetrics2, logbr2, samples)

    # Calculate the integral using the trapezoid method on the samples.
    int1 = numpy.trapz(v1, dx=interval)
    int2 = numpy.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2 - int1) / (max_int - min_int)
    avg_diff = (math.exp(avg_exp_diff) - 1) * 100

    return avg_diff

'''
if __name__ == "__main__":
    brs1 = [64052.664, 6468.096, 4673.424, 3179.4, 2298.384, 1361.184]
    qtys1 = [1, 1, 0.99999, 0.99998, 0.99996, 0.99992]
    brs2 = [68461.896, 7554.96, 4827.432, 3294.024, 2380.128, 1401.744]
    qtys2 = [1, 1, 0.99999, 0.99998, 0.99996, 0.99992]

    bdrate = BD_RATE(brs1, qtys1, brs2, qtys2)
    print("bdrate calculated is %3.3f%%" % bdrate)
'''
