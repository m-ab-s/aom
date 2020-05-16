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

######################################
# configuration settings
######################################
RootPath = "..\\"
BinPath = os.path.join(RootPath, 'bin')
WorkPath = os.path.join(RootPath, 'test')
SMOKE_TEST = False  # override some parameters to do a quick smoke test
FrameNum = 60

if SMOKE_TEST:
    FrameNum = 2

############ test contents #######################################
ContentPath = "..\\video"
# when Have_Class_Subfolder is set to True, test clips are put in subfolders
# under ContentPath, each subfolder name represents a class. Class name in the
# table of Clips is the subfolder name
Have_Class_Subfolder = False
Clips = {
#    basename           class      width   height  framerate   bitdepth    fmt
    #"CrowdRun":         ["ClassB",  1920,   1080,   30,     8,      "yuv420p"],
    #"BasketballDrive":  ["ClassB",  1920,   1080,   30,     8,      "yuv420p"],
    "NetflixCrosswalk_1920x1080_60fps_8bit_420_60f":    ["ClassB",  1920,   1080,
     30,         8,          "yuv420p"],
}
'''
    "aspen_1080p_60f":  ["ClassB",  1920,   1080,   30,         8,
              "yuv420p"],
    "dark720p_60f":     ["ClassB",  1280,   720,    30,         8,
              "yuv420p"],
    "DOTA2_60f_420":    ["ClassB",  1920,   1080,   30,         8,
              "yuv420p"],
    "duckstakeoff_1080p50_60f":     ["ClassB",  1920,   1080,   30,         8,
              "yuv420p"],
    "KristenAndSara_1280x720_60f":  ["ClassB",  1280,   720,    30,         8,
              "yuv420p"],
    "life_1080p30_60f":             ["ClassB",  1920,   1080,   30,         8,
              "yuv420p"],
    "MINECRAFT_60f_420":            ["ClassB",  1920,   1080,   30,         8,
              "yuv420p"],
    "NetflixAerial_1920x1080_60fps_8bit_420_60f":       ["ClassB",  1920,   1080,
       30,         8,          "yuv420p"],
    "NetflixBoat_1920x1080_60fps_8bit_420_60f":         ["ClassB",  1920,   1080,
       30,         8,          "yuv420p"],
    "NetflixCrosswalk_1920x1080_60fps_8bit_420_60f":    ["ClassB",  1920,   1080,
       30,         8,          "yuv420p"],
    "NetflixDrivingPOV_1280x720_60fps_8bit_420_60f":    ["ClassB",  1280,   720,
       30,         8,          "yuv420p"],
    "NetflixFoodMarket_1920x1080_60fps_8bit_420_60f":   ["ClassB",  1920,   1080,
       30,         8,          "yuv420p"],
    "NetflixPierSeaside_1920x1080_60fps_8bit_420_60f":  ["ClassB",  1920,   1080,
       30,         8,          "yuv420p"],
    "NetflixRollerCoaster_1280x720_60fps_8bit_420_60f": ["ClassB",  1280,   720,
        30,         8,          "yuv420p"],
    "NetflixSquareAndTimelapse_1920x1080_60fps_8bit_420_60f":   ["ClassB",  1920,
       1080,   30,         8,          "yuv420p"],
    "NetflixTunnelFlag_1920x1080_60fps_8bit_420_60f":           ["ClassB",  1920,
       1080,   30,         8,          "yuv420p"],
    "rushhour_1080p25_60f": ["ClassB",  1920,   1080,   30,         8,
              "yuv420p"],
    "STARCRAFT_60f_420":    ["ClassB",  1920,   1080,   30,         8,
              "yuv420p"],
    "touchdownpass_1080p_60f":  ["ClassB",  1920,   1080,   30,         8,
              "yuv420p"],
    "vidyo1_720p_60fps_60f":    ["ClassB",  1280,   720,    30,         8,
              "yuv420p"],
    "vidyo4_720p_60fps_60f":    ["ClassB",  1280,   720,    30,         8,
              "yuv420p"],
    "wikipedia_420":            ["ClassB",  1920,   1080,   30,         8,
              "yuv420p"],
'''
############## Scaling settings ############################################
# down scaling ratio
DnScaleRatio = [1.0, 1.5, 2.0, 3.0, 4.0] #, 6.0]  # downscale ratio
#down and up scaling algorithm, the 2 lists should be of same size
DnScalingAlgos = ['bicubic', 'bilinear', 'gauss', 'lanczos', 'sinc']
UpScalingAlgos = ['bicubic', 'bilinear', 'gauss', 'lanczos', 'sinc']

if SMOKE_TEST:
    DnScalingAlgos = ['bicubic', 'lanczos', 'sinc']
    UpScalingAlgos = ['bicubic', 'lanczos', 'sinc']

##################### Encode Config ########################################
EncodeMethods = ["ffmpeg", "aom", "svt"]
CodecNames = ["hevc", "av1"]
SUFFIX = {"hevc": ".265", "av1": ".ivf"}
FFMPEG = os.path.join(BinPath, 'ffmpeg.exe')
AOMENC = os.path.join(BinPath, 'aomenc.exe')
SVTAV1 = os.path.join(BinPath, 'SvtAv1EncApp.exe')
SVTHEVC = os.path.join(BinPath, 'SvtHevcEncApp.exe')
AOMDEC = os.path.join(BinPath, 'aomdec.exe')

QPs = list(range(22, 51, 5))  # total 6 QPs
if SMOKE_TEST:
    QPs = list(range(12, 45, 10))  # total 4 QPs

######################## quality evalution config #############################
QualityList = ['VMAF_Y', 'PSNR_Y', 'PSNR_U', 'PSNR_V', 'SSIM_Y', 'MS-SSIM_Y']
# method should be one of 'VMAF', 'FFMPEG' or 'HDRTools'
QualityEvalMethods = ['VMAF', 'HDRTools', 'HDRTools', 'HDRTools', 'HDRTools',
                      'HDRTools']
VMAF = os.path.join(BinPath, 'vmafossexec.exe')
HDRTool = os.path.join(BinPath, 'HDRMetrics.exe')

######################## config for exporting data to excel  #################
#https://xlsxwriter.readthedocs.io/working_with_colors.html#colors
# line color used, number of colors >= len(DnScaledRes)
LineColors = ['blue', 'red', 'green', 'orange', 'pink', 'yellow']
ConvexHullColor = 'white'

# format for exported excel of convexhull test
# if to modify below 3 variables, need to modify function
# SaveConvexHullResultsToExcel accordingly
CvxH_startCol = 1; CvxH_startRow = 2; CvxH_colInterval = 2
CvxH_WtCols = [(CvxH_colInterval + 1 + len(QualityList)) * i + CvxH_startCol
               for i in range(len(DnScaleRatio))]
CvxH_WtRows = [CvxH_startRow + i for i in range(len(QPs))]
CvxH_WtLastCol = CvxH_WtCols[-1] + len(QualityList)
CvxH_WtLastRow = CvxH_WtRows[-1]
######################## post analysis #########################################
PostAnalysis_Path = os.path.join(RootPath, 'analysis')
Path_RDResults = os.path.join(PostAnalysis_Path, 'rdresult')
SummaryOutPath = os.path.join(PostAnalysis_Path, 'summary')
Path_ScalingResults = os.path.join(PostAnalysis_Path, 'scalingresult')
# vba file needed when to calculate bdrate
#VbaBinFile = os.path.join(BinPath, 'vbaProject_JVET-L0242.bin')
VbaBinFile = os.path.join(BinPath, 'vbaProject-AV2.bin')

# format for exported excel of scaling quality test
# if to modify below 3 variables, need to modify function SaveScalingResultsToExcel
# accordingly
ScalQty_startCol = 6; ScalQty_startRow = 2; ScalQty_colInterval = 1
ScalSumQty_startCol = 7
ScalQty_WtCols = [(ScalQty_colInterval +
                   len(QualityList)) * i + ScalQty_startCol
                  for i in range(len(DnScalingAlgos))]
ScalSumQty_WtCols = [(ScalQty_colInterval +
                      len(QualityList)) * i + ScalQty_startCol + 1
                     for i in range(len(DnScalingAlgos))]
######################## logging #########################################
LoggerName = "ConvexHullTest"
LogLevels = ['NONE', 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
LogCmdOnly = False
