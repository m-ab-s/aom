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
from Config import BinPath, LoggerName, VMAF
from Utils import GetShortContentName, ExecuteCmd

subloggername = "CalcQtyMetrics_VMAFTool"
loggername = LoggerName + '.' + '%s' % subloggername
logger = logging.getLogger(loggername)

Model_Pkg_File = os.path.join(BinPath, 'vmaf_v0.6.1.pkl')
VMAFMetricsFullList = ['VMAF_Y','VMAF_Y-NEG','PSNR_Y','PSNR_U','PSNR_V','SSIM_Y(dB)',
                       'MS-SSIM_Y(dB)','PSNR-HVS','CIEDE2000','APSNR_Y',
                       'APSNR_U','APSNR_V']

def ParseVMAFLogFile(vmaf_log):
    floats = len(VMAFMetricsFullList) * [0.0]
    flog = open(vmaf_log, 'r')
    for line in flog:
        m = re.search(r"\"vmaf\".*\s+mean=\"(\d+\.?\d*)\"\s+",line)
        if m:
            floats[0] = m.group(1)
        m = re.search(r"\"vmaf_neg\".*\s+mean=\"(\d+\.?\d*)\"\s+", line)
        if m:
            floats[1] = m.group(1)
        m = re.search(r"\"psnr_y\".*\s+mean=\"(\d+\.?\d*)\"\s+", line)
        if m:
            floats[2] = m.group(1)
        m = re.search(r"\"psnr_cb\".*\s+mean=\"(\d+\.?\d*)\"\s+", line)
        if m:
            floats[3] = m.group(1)
        m = re.search(r"\"psnr_cr\".*\s+mean=\"(\d+\.?\d*)\"\s+", line)
        if m:
            floats[4] = m.group(1)
        m = re.search(r"\"float_ssim\".*\s+mean=\"(\d+\.?\d*)\"\s+", line)
        if m:
            floats[5] = m.group(1)
        m = re.search(r"\"float_ms_ssim\".*\s+mean=\"(\d+\.?\d*)\"\s+", line)
        if m:
            floats[6] = m.group(1)
        m = re.search(r"\"psnr_hvs\".*\s+mean=\"(\d+\.?\d*)\"\s+", line)
        if m:
            floats[7] = m.group(1)
        m = re.search(r"\"ciede2000\".*\s+mean=\"(\d+\.?\d*)\"\s+", line)
        if m:
            floats[8] = m.group(1)
        #<aggregate_metrics apsnr_y="46.817276" apsnr_cb="49.092538" apsnr_cr="50.014785" />
        m = re.search(r"aggregate_metrics\s+apsnr_y=\"(\d+\.?\d*)\"\s+apsnr_cb=\"(\d+\.?\d*)\"\s+apsnr_cr=\"(\d+\.?\d*)\"", line)
        if m:
            floats[9] = m.group(1)
            floats[10] = m.group(2)
            floats[11] = m.group(3)
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
def VMAF_CalQualityMetrics(origfile, recfile, fmt, num, w, h, bit_depth,
                           logfilePath, LogCmdOnly=False):
    vmaf_log = GetVMAFLogFile(recfile, logfilePath)
    args = " -r %s -d %s --aom_ctc v1.0 -q  -o %s" \
           % (origfile, recfile, vmaf_log)
    cmd = VMAF + args
    ExecuteCmd(cmd, LogCmdOnly)

def VMAF_GatherQualityMetrics(recfile, logfilePath):
    vmaf_log = GetVMAFLogFile(recfile, logfilePath)
    results = ParseVMAFLogFile(vmaf_log)
    return results
