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
from Config import BinPath, LoggerName, LogCmdOnly, FFMPEG
from Utils import GetShortContentName, ExecuteCmd

subloggername = "CalcQtyMetrics_FFMPEGTool"
loggername = LoggerName + '.' + '%s' % subloggername
logger = logging.getLogger(loggername)


FFMPEGMetricsFullList = ['PSNR_Y', 'PSNR_U', 'PSNR_V']

def ParseFfmpegLogFile(psnr_log):
    floats = len(FFMPEGMetricsFullList) * [0.0]
    flog = open(psnr_log, 'r')
    cnt = 0
    for line in flog:
        cnt += 1
        item = re.findall(r"psnr_y:(\d+\.?\d*)", line)
        floats[0] += 0 if len(item) == 0 else float(item[0])
        item = re.findall(r"psnr_u:(\d+\.?\d*)", line)
        floats[1] += 0 if len(item) == 0 else float(item[0])
        item = re.findall(r"psnr_v:(\d+\.?\d*)", line)
        floats[2] += 0 if len(item) == 0 else float(item[0])

    floats = [float(i) / cnt for i in floats]

    print_str = "FFMPEG quality metrics: "
    for metrics, idx in zip(FFMPEGMetricsFullList, range(len(FFMPEGMetricsFullList))):
        print_str += "%s = %2.5f, " % (metrics, floats[idx])
    logger.info(print_str)

    return floats[0:len(FFMPEGMetricsFullList)]


def GetFfmpegLogFile(recfile, path):
    filename = GetShortContentName(recfile, False) + '_psnr.log'
    file = os.path.join(path, filename)
    return file

################################################################################
##################### Exposed Functions ########################################
def FFMPEG_CalQualityMetrics(origfile, recfile, num, w, h, logfilePath):
    #calculate psnr using ffmpeg filter to get psnr_u and psnr_v
    #here we have to pass the psnr_log file name to FFMPEG without the target
    #log path first, then copy the result psnr log to the target log path after
    #the ffmpeg call. it doesn't work if the full path is directly passed to FFMPEG
    psnr_log = GetFfmpegLogFile(recfile, BinPath)
    psnr_log = os.path.basename(psnr_log)
    args = " -s %dx%d -pix_fmt yuv420p -i %s -s %dx%d -pix_fmt yuv420p -i %s" \
           " -frames:v %d -lavfi psnr=%s -f null -" % \
           (w, h, origfile, w, h, recfile, num, psnr_log)
    cmd = FFMPEG + args
    ExecuteCmd(cmd, LogCmdOnly)
    #move the psnr log to the target log path
    target = os.path.join(logfilePath, psnr_log)
    cmd = "move %s %s" % (psnr_log, target)
    ExecuteCmd(cmd, LogCmdOnly)

def FFMPEG_GatherQualityMetrics(recfile, logfilePath):
    psnr_log = GetFfmpegLogFile(recfile, logfilePath)
    results = ParseFfmpegLogFile(psnr_log)
    return results
