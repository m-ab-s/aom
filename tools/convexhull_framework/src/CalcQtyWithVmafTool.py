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

import os
import re
import logging
from Config import BinPath, LoggerName, LogCmdOnly, VMAF
from Utils import GetShortContentName, ExecuteCmd

subloggername = "CalcQtyMetrics_VMAFTool"
loggername = LoggerName + '.' + '%s' % subloggername
logger = logging.getLogger(loggername)

Model_Pkg_File = os.path.join(BinPath, 'vmaf_v0.6.1.pkl')
VMAFMetricsFullList = ['VMAF_Y', 'PSNR_Y', 'SSIM_Y', 'MS-SSIM_Y']

def ParseVMAFLogFile(vmaf_log):
    floats = len(VMAFMetricsFullList) * [0.0]
    flog = open(vmaf_log, 'r')
    for line in flog:
        if 'aggregateVMAF' in line:
            item = re.findall(r"aggregateVMAF=\"(\d+\.?\d*)\"", line)
            floats[0] = 0 if len(item) == 0 else item[0]
            item = re.findall(r"aggregatePSNR=\"(\d+\.?\d*)\"", line)
            floats[1] = 0 if len(item) == 0 else item[0]
            item = re.findall(r"aggregateSSIM=\"(\d+\.?\d*)\"", line)
            floats[2] = 0 if len(item) == 0 else item[0]
            item = re.findall(r"aggregateMS_SSIM=\"(\d+\.?\d*)\"", line)
            floats[3] = 0 if len(item) == 0 else item[0]
            break
    flog.close()

    floats = [float(i) for i in floats]

    print_str = "VMAF quality metrics: "
    for metrics, idx in zip(VMAFMetricsFullList, range(len(VMAFMetricsFullList))):
        print_str += "%s = %2.5f, " % (metrics, floats[idx])
    logger.info(print_str)

    return floats[0:len(VMAFMetricsFullList)]


def GetVMAFLogFile(recfile, path):
    filename = GetShortContentName(recfile, False) + '_vmaf.log'
    file = os.path.join(path, filename)
    return file

################################################################################
##################### Exposed Functions ########################################
def VMAF_CalQualityMetrics(origfile, recfile, num, w, h, logfilePath):
    vmaf_log = GetVMAFLogFile(recfile, logfilePath)
    args = " yuv420p %d %d %s %s %s --log %s --psnr --ssim --ms-ssim "\
           % (w, h, origfile, recfile, Model_Pkg_File, vmaf_log)
    cmd = VMAF + args
    ExecuteCmd(cmd, LogCmdOnly)

def VMAF_GatherQualityMetrics(recfile, logfilePath):
    vmaf_log = GetVMAFLogFile(recfile, logfilePath)
    results = ParseVMAFLogFile(vmaf_log)
    return results
