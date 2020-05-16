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
from VideoEncoder import VideoEncode
from VideoDecoder import VideoDecode
from VideoScaler import UpScaling, GetDownScaledOutFile, GetUpScaledOutFile
from Config import SUFFIX, LoggerName
from Utils import GetShortContentName
import logging

subloggername = "EncDecUpscale"
loggername = LoggerName + '.' + '%s' % subloggername
logger = logging.getLogger(loggername)

################################################################################
##################### Internal Helper Functions ################################
def GetBitstreamFile(method, codec, preset, yuvfile, qp, outpath):
    bs_suffix = SUFFIX[codec]
    Prefix_EncodeCfg = '_%s_%s_Preset_%s' % (method, codec, preset)
    filename = GetShortContentName(yuvfile, False) + Prefix_EncodeCfg + "_QP_"\
               + str(qp) + bs_suffix
    filename = os.path.join(outpath, filename)
    return filename

def GetDecodedFile(bsfile, outpath):
    filename = GetShortContentName(bsfile, False) + '_Decoded.yuv'
    decodedfile = os.path.join(outpath, filename)
    return decodedfile

################################################################################
##################### Major Functions ##########################################
def Encode(method, codec, preset, input, qp, num, framerate, width, height, path):
    bsfile = GetBitstreamFile(method, codec, preset, input, qp, path)
    # call VideoEncoder to do the encoding
    VideoEncode(method, codec, input, qp, num, framerate, width, height, bsfile,
                preset)
    return bsfile


def Decode(codec, bsfile, path):
    decodedfile = GetDecodedFile(bsfile, path)
    #call VideoDecoder to do the decoding
    VideoDecode(codec, bsfile, decodedfile)
    return decodedfile


def Run_EncDec_Upscale(method, codec, preset, input, QP, num, framerate, inw,
                       inh, outw, outh, path_bs, path_decoded,
                       path_upscaled, algo):
    logger.info("%s %s start encode file %s with QP = %d"
                % (method, codec, os.path.basename(input), QP))
    bsFile = Encode(method, codec, preset, input, QP, num, framerate, inw, inh,
                    path_bs)
    logger.info("start decode file %s" % os.path.basename(bsFile))
    decodedYUV = Decode(codec, bsFile, path_decoded)
    logger.info("start upscale file %s" % os.path.basename(decodedYUV))
    upscaledYUV = UpScaling(decodedYUV, num, inw, inh, outw, outh, path_upscaled,
                            algo)
    logger.info("finish Run Encode, Decode and Upscale")
    return upscaledYUV


def GetBsReconFileName(encmethod, codecname, preset, content, width, height,
                       dnwidth, dnheight, dnScAlgo,
                       upScAlgo, qp, path_bs):
    dsyuv_name = GetDownScaledOutFile(content, width, height, dnwidth, dnheight,
                                      path_bs, dnScAlgo)
    # return bitstream file with absolute path
    bs = GetBitstreamFile(encmethod, codecname, preset, dsyuv_name, qp, path_bs)
    decoded = GetDecodedFile(bs, path_bs)
    reconfilename = GetUpScaledOutFile(decoded, dnwidth, dnheight, width, height,
                                       path_bs, upScAlgo)
    # return only Recon yuv file name w/o path
    reconfilename = GetShortContentName(reconfilename, False)
    return bs, reconfilename
