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

import logging
from Config import QualityList, LoggerName, QualityEvalMethods
import Utils
from CalcQtyWithVmafTool import VMAF_CalQualityMetrics, VMAF_GatherQualityMetrics,\
     VMAFMetricsFullList
from CalcQtyWithHdrTools import HDRTool_CalQualityMetrics, HDRToolsMetricsFullList,\
     HDRTool_GatherQualityMetrics
from CalcQtyWithFfmpeg import FFMPEG_GatherQualityMetrics, FFMPEGMetricsFullList,\
     FFMPEG_CalQualityMetrics

subloggername = "CalcQtyMetrics"
loggername = LoggerName + '.' + '%s' % subloggername
logger = logging.getLogger(loggername)

MetricsFullDict = {'VMAF': VMAFMetricsFullList, 'HDRTools': HDRToolsMetricsFullList,
                   'FFMPEG': FFMPEGMetricsFullList}

def CalculateQualityMetric(content, framenum, reconYUV, width, height, logfilePath,
                           cfgfilePath):
    Utils.CmdLogger.write("::Quality Metrics\n")

    methods_torun = list(set(QualityEvalMethods))  # remove duplicate items
    for method in methods_torun:
        if method == 'VMAF':
            VMAF_CalQualityMetrics(content, reconYUV, framenum, width, height,
                                   logfilePath)
        elif method == 'HDRTools':
            HDRTool_CalQualityMetrics(content, reconYUV, framenum, width, height,
                                      logfilePath, cfgfilePath)
        elif method == 'FFMPEG':
            FFMPEG_CalQualityMetrics(content, reconYUV, framenum, width, height,
                                     logfilePath)
        else:
            logger.error("invalid quality evaluation method: %s !" % method)
            return

def GatherQualityMetrics(reconYUV, logfilePath):
    methods_torun = list(set(QualityEvalMethods))  # remove duplicate items
    qresult_dict = {}
    for method in methods_torun:
        if method == 'VMAF':
            qresult_dict[method] = VMAF_GatherQualityMetrics(reconYUV, logfilePath)
        elif method == 'HDRTools':
            qresult_dict[method] = HDRTool_GatherQualityMetrics(reconYUV, logfilePath)
        elif method == 'FFMPEG':
            qresult_dict[method] = FFMPEG_GatherQualityMetrics(reconYUV, logfilePath)
        else:
            logger.error("invalid quality evaluation method: %s !" % method)
            return

    results = []
    for metric, method in zip(QualityList, QualityEvalMethods):
        mfullList = MetricsFullDict[method]
        if metric in mfullList:
            indx = mfullList.index(metric)
            results.append(qresult_dict[method][indx])
        else:
            logger.error("invalid quality metrics in QualityList")
            results.append('none')

    return results
