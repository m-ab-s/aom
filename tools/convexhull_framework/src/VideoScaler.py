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
import Utils
import logging
from Config import BinPath, LogCmdOnly, LoggerName, FFMPEG
from Utils import GetShortContentName, ExecuteCmd

subloggername = "VideoScaler"
loggername = LoggerName + '.' + '%s' % subloggername
logger = logging.getLogger(loggername)

def ValidAlgo_ffmpeg(algo):
    if (algo == 'bicubic' or algo == 'lanczos' or algo == 'sinc' or
        algo == 'bilinear' or algo == 'spline' or algo == 'gauss' or
        algo == 'bicublin' or algo == 'neighbor'):
        return True
    else:
        return False

#use ffmpeg to do image rescaling for yuv420 8bit
def RescaleWithFfmpeg(infile, inw, inh, outw, outh, algo, outfile, num, app_path):
    args = " -y -s:v %dx%d -i %s -vf scale=%dx%d -c:v rawvideo -pix_fmt yuv420p" \
           " -sws_flags %s+accurate_rnd+full_chroma_int -sws_dither none" \
           % (inw, inh, infile, outw, outh, algo)
    if (algo == 'lanczos'):
        args += " -param0 5 "
    args += " -frames %d %s" % (num, outfile)
    cmd = FFMPEG + args
    ExecuteCmd(cmd, LogCmdOnly)

def VideoRescaling(infile, num, inw, inh, outw, outh, outfile, algo):
    if ValidAlgo_ffmpeg(algo):
        RescaleWithFfmpeg(infile, inw, inh, outw, outh, algo, outfile, num, BinPath)
    # add other tools for scaling here later
    else:
        logger.error("unsupported scaling algorithm.")

####################################################################################
##################### Major Functions ################################################
def GetDownScaledOutFile(input, inw, inh, dnw, dnh, path, algo):
    contentBaseName = GetShortContentName(input)
    actual_algo = 'None' if inw == dnw and inh == dnh else algo
    filename = contentBaseName + ('_DnScaled_%s_%dx%d.yuv' % (actual_algo, dnw,
                                                              dnh))
    dnscaledout = os.path.join(path, filename)
    return dnscaledout

def GetUpScaledOutFile(infile, inw, inh, outw, outh, path, algo):
    actual_algo = 'None' if inw == outw and inh == outh else algo
    filename = GetShortContentName(infile, False) + ('_UpScaled_%s_%dx%d.yuv'
                                                     % (actual_algo, outw, outh))
    upscaledout = os.path.join(path, filename)
    return upscaledout

def DownScaling(input, num, inw, inh, outw, outh, path, algo):
    dnScalOut = GetDownScaledOutFile(input, inw, inh, outw, outh, path, algo)

    Utils.CmdLogger.write("::Downscaling\n")
    if (inw == outw and inh == outh):
        cmd = "copy %s %s" % (input, dnScalOut)
        ExecuteCmd(cmd, LogCmdOnly)
    else:
        # call separate process to do the downscaling
        VideoRescaling(input, num, inw, inh, outw, outh, dnScalOut, algo)
    return dnScalOut

def UpScaling(infile, num, inw, inh, outw, outh, path, algo):
    upScaleOut = GetUpScaledOutFile(infile, inw, inh, outw, outh, path, algo)

    Utils.CmdLogger.write("::Upscaling\n")
    if (inw == outw and inh == outh):
        cmd = "copy %s %s" % (infile, upScaleOut)
        ExecuteCmd(cmd, LogCmdOnly)
    else:
        # call separate process to do the upscaling
        VideoRescaling(infile, num, inw, inh, outw, outh, upScaleOut, algo)
    return upScaleOut
