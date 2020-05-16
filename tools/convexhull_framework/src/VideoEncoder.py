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

import Utils
from Config import BinPath, LogCmdOnly, FFMPEG, AOMENC, SVTAV1, SVTHEVC
from Utils import ExecuteCmd

#use ffmpeg to encode video
def EncodeWithFfmpeg_HEVC(infile, QP, num,fr, width, height, outfile, preset):
    EncodeProfile = 'main'
    args = " -y -s %dx%d -pix_fmt yuv420p  -r %s -i %s -frames:v %d -g %d -bf 7" \
           " -bsf:v hevc_mp4toannexb -c:v libx265 -preset %s -profile:v %s " \
           "-x265-params \"qp=%d:aq-mode=0:b-adapt=0:bframes=7:b-pyramid=1:" \
           "no-scenecut=1:no-open-gop=1:input-depth=8:output-depth=8\" %s" % (
            width, height, fr, infile, num, int(2 * fr),  # gop size = 2 seconds
            preset, EncodeProfile, (QP+3), outfile)
    cmd = FFMPEG + args
    ExecuteCmd(cmd, LogCmdOnly)

def EncodeWithAOM_AV1(infile, QP, framenum, framerate, width, height, outfile,
                      preset):
    args = " --codec=av1 -v --psnr --ivf  --frame-parallel=0 --cpu-used=%s" \
           " --limit=%d --auto-alt-ref=1 --passes=2  " \
           "--threads=1 --lag-in-frames=25 --end-usage=q --cq-level=%d" \
           " -w %d -h %d -o %s %s"\
           % (preset, framenum, QP, width, height, outfile, infile)
    cmd = AOMENC + args
    ExecuteCmd(cmd, LogCmdOnly)

def EncodeWithSVT_AV1(infile, QP, framenum, framerate, width, height, outfile,
                      preset):
    args = " --preset %s --scm 2 --lookahead 0 --hierarchical-levels 3 -n %d" \
           " --keyint 255 -rc 0 -q %d -w %d -h %d -b %s -i %s"\
           % (str(preset), framenum, QP, width, height, outfile, infile)
    cmd = SVTAV1 + args
    ExecuteCmd(cmd, LogCmdOnly)

def EncodeWithSVT_HEVC(infile, QP, framenum, framerate, width, height, outfile,
                       preset):
    args = " -i %s  -w %d -h %d -encMode %s -hierarchical-levels 3" \
           " -intra-period 100 -scd 0 -rc 0 -q %d -n %d -b %s"\
           % (infile, width, height, preset, QP, framenum, outfile)
    cmd = SVTHEVC + args
    ExecuteCmd(cmd, LogCmdOnly)

def VideoEncode(EncodeMethod, CodecName, infile, QP, framenum, framerate, width,
                height, outfile, preset):
    Utils.CmdLogger.write("::Encode\n")
    if EncodeMethod == "ffmpeg":
        if CodecName == 'hevc':
            EncodeWithFfmpeg_HEVC(infile, QP, framenum, framerate, width, height,
                                  outfile, preset)
        else:
            raise ValueError("invalid parameter for encode.")
    elif EncodeMethod == "aom":
        if CodecName == 'av1':
            EncodeWithAOM_AV1(infile, QP, framenum, framerate, width, height,
                              outfile, preset)
        else:
            raise ValueError("invalid parameter for encode.")
    elif EncodeMethod == "svt":
        if CodecName == 'av1':
            EncodeWithSVT_AV1(infile, QP, framenum, framerate, width, height,
                              outfile, preset)
        elif CodecName == 'hevc':
            EncodeWithSVT_HEVC(infile, QP, framenum, framerate, width, height,
                               outfile, preset)
        else:
            raise ValueError("invalid parameter for encode.")
    else:
        raise ValueError("invalid parameter for encode.")
