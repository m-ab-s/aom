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

from shutil import copyfile
import fileinput
import os
import re
from Utils import GetShortContentName, ExecuteCmd
from Config import BinPath, LogCmdOnly, LoggerName, HDRTool
import logging

subloggername = "HDRToolsRun"
loggername = LoggerName + '.' + '%s' % subloggername
logger = logging.getLogger(loggername)


HDRToolsConfigFileTemplate = os.path.join(BinPath, 'HDRMetricsYUV420_template.cfg')
HDRToolsMetricsFullList = ['PSNR_Y', 'PSNR_U', 'PSNR_V', 'MS-SSIM_Y', 'MS-SSIM_U',
                           'MS-SSIM_V', 'SSIM_Y', 'SSIM_U', 'SSIM_V']

def GenerateCfgFile(orig_file, rec_file, num, w, h, configpath):
    filename = GetShortContentName(rec_file, False) + '_hdrtools.cfg'
    cfgfile = os.path.join(configpath, filename)
    copyfile(HDRToolsConfigFileTemplate, cfgfile)
    fp = fileinput.input(cfgfile, inplace=1)
    for line in fp:
        if 'Input0File=' in line:
            line = 'Input0File="%s"\n' % orig_file
        if 'Input1File=' in line:
            line = 'Input1File="%s"\n' % rec_file
        if 'NumberOfFrames=' in line:
            line = 'NumberOfFrames=%d\n' % num
        if 'Input0Width=' in line:
            line = 'Input0Width=%d\n' % w
        if 'Input1Width=' in line:
            line = 'Input1Width=%d\n' % w
        if 'Input0Height=' in line:
            line = 'Input0Height=%d\n' % h
        if 'Input1Height=' in line:
            line = 'Input1Height=%d\n' % h
        print(line, end='')
    fp.close()
    return cfgfile

def ParseHDRToolLogFile(logfile):
    flog = open(logfile, 'r')
    pattern = re.compile(r'\d+\.?\d*')
    floats = len(HDRToolsMetricsFullList) * [0.0]
    for line in flog:
        if 'D_Avg' in line:
            floats = [float(i) for i in pattern.findall(line)]
            break

    print_str = "HDRTools qty metrics: "
    for metrics, idx in zip(HDRToolsMetricsFullList,
                            range(len(HDRToolsMetricsFullList))):
        print_str += "%s = %2.5f, " % (metrics, floats[idx])
    logger.info(print_str)
    return floats[0:len(HDRToolsMetricsFullList)]

def GetHDRLogFile(recfile, path):
    filename = GetShortContentName(recfile, False) + '_hdrtools.log'
    file = os.path.join(path, filename)
    return file

################################################################################
##################### Exposed Functions ########################################
def HDRTool_CalQualityMetrics(origfile, recfile, num, w, h, logfilepath,
                              cfgfilepath):
    cfgFile = GenerateCfgFile(origfile, recfile, num, w, h, cfgfilepath)
    logfile = GetHDRLogFile(recfile, logfilepath)
    args = " -f %s > %s" % (cfgFile, logfile)
    cmd = HDRTool + args
    ExecuteCmd(cmd, LogCmdOnly)

def HDRTool_GatherQualityMetrics(recfile, logfilepath):
    logfile = GetHDRLogFile(recfile, logfilepath)
    results = ParseHDRToolLogFile(logfile)
    return results
