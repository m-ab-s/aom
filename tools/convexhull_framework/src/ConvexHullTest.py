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
import sys
import xlsxwriter
import argparse
from EncDecUpscale import Run_EncDec_Upscale, GetBsReconFileName
from VideoScaler import GetDownScaledOutFile, DownScaling
from CalculateQualityMetrics import CalculateQualityMetric, GatherQualityMetrics
from Utils import GetShortContentName, GetVideoInfo, CreateChart_Scatter,\
     AddSeriesToChart_Scatter, InsertChartsToSheet, CreateNewSubfolder,\
     GetContents, SetupLogging, UpdateChart, AddSeriesToChart_Scatter_Rows,\
     Cleanfolder
from PostAnalysis_Summary import GenerateSummaryExcelFile,\
     GenerateSummaryConvexHullExcelFile
from ScalingTest import Run_Scaling_Test, SaveScalingResultsToExcel
import Utils
from Config import LogLevels, FrameNum, DnScaleRatio, QPs, CvxH_WtCols,\
     CvxH_WtRows, QualityList, LineColors, DnScalingAlgos, UpScalingAlgos,\
     ContentPath, SummaryOutPath, WorkPath, Path_RDResults, Clips, \
     ConvexHullColor, EncodeMethods, CodecNames, LoggerName, LogCmdOnly

###############################################################################
##### Helper Functions ########################################################
def CleanIntermediateFiles():
    folders = [Path_DecodedYuv, Path_CfgFiles]
    if not KeepUpscaledOutput:
        folders += [Path_DecUpScaleYuv, Path_UpScaleYuv]

    for folder in folders:
        Cleanfolder(folder)

def GetRDResultExcelFile(content):
    contentBaseName = GetShortContentName(content)
    filename = "RDResults_%s_%s_%s_%s.xlsx" % (contentBaseName, EncodeMethod,
                                               CodecName, EncodePreset)
    file = os.path.join(Path_RDResults, filename)
    return file

def setupWorkFolderStructure():
    global Path_Bitstreams, Path_DecodedYuv, Path_UpScaleYuv, Path_DnScaleYuv,\
           Path_QualityLog, Path_TestLog, Path_CfgFiles, Path_DecUpScaleYuv
    Path_Bitstreams = CreateNewSubfolder(WorkPath, "bistreams")
    Path_DecodedYuv = CreateNewSubfolder(WorkPath, "decodedYUVs")
    Path_UpScaleYuv = CreateNewSubfolder(WorkPath, "upscaledYUVs")
    Path_DecUpScaleYuv = CreateNewSubfolder(WorkPath, "decUpscaledYUVs")
    Path_DnScaleYuv = CreateNewSubfolder(WorkPath, "downscaledYUVs")
    Path_QualityLog = CreateNewSubfolder(WorkPath, "qualityLogs")
    Path_TestLog = CreateNewSubfolder(WorkPath, 'testLogs')
    Path_CfgFiles = CreateNewSubfolder(WorkPath, "configFiles")

'''
The convex_hull function is adapted based on the original python implementation
from https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
It is changed to return the lower and upper portions of the convex hull separately
to get the convex hull based on traditional rd curve, only the upper portion is
needed.
'''

def convex_hull(points):
    """Computes the convex hull of a set of 2D points.
    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross
    # product. Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower, upper

def AddConvexHullCurveToCharts(sht, startrow, startcol, charts, rdPoints,
                               dnScaledRes, sum_sht, sum_start_row):
    shtname = sht.get_name()
    sht.write(startrow, startcol, "ConvexHull Data")
    numitems = 5  # qty, bitrate, qp, resolution, 1 empty row as internal
    rows = [startrow + 1 + numitems * i for i in range(len(QualityList))]

    hull = {}
    max_len = 0

    for qty, idx, row in zip(QualityList, range(len(QualityList)), rows):
        lower, upper = convex_hull(rdPoints[idx])
        hull[qty] = upper
        max_len = max(max_len, len(upper))
        sht.write(row, startcol, qty)
        sht.write(row + 1, startcol, "Bitrate(kbps)")
        sht.write(row + 2, startcol, "QP")
        sht.write(row + 3, startcol, 'Resolution')

        brts = [h[0] for h in hull[qty]]
        qtys = [h[1] for h in hull[qty]]
        sht.write_row(row, startcol + 1, qtys)
        sht.write_row(row + 1, startcol + 1, brts)

        rdpnts_qtys = [rd[1] for rd in rdPoints[idx]]
        cvh_qidxs = [rdpnts_qtys.index(qty) for qty in qtys]
        cvh_QPs = [QPs[i % len(QPs)] for i in cvh_qidxs]
        cvh_Res = [dnScaledRes[i // len(QPs)] for i in cvh_qidxs]
        cvh_Res_txt = ["%sx%s" % (x, y) for (x, y) in cvh_Res]
        sht.write_row(row + 2, startcol + 1, cvh_QPs)
        sht.write_row(row + 3, startcol + 1, cvh_Res_txt)

        cols = [startcol + 1 + i for i in range(len(hull[qty]))]
        AddSeriesToChart_Scatter_Rows(shtname, cols, row, row + 1, charts[idx],
                                      'ConvexHull', ConvexHullColor)

    endrow = rows[-1] + numitems

    #add convex hull data into summary sheet. the length of each convex hull
    #could be different
    sum_row = sum_start_row
    sum_sht.write(sum_row, 2, max_len)
    sum_col = 3
    for qty in QualityList:
        row = sum_row
        for point in hull[qty]:
            sum_sht.write(row, sum_col, point[0])
            sum_sht.write(row, sum_col + 1, point[1])
            row += 1
        sum_col += 2

    return endrow, sum_start_row + max_len - 1

###############################################################################
######### Major Functions #####################################################
def CleanUp_workfolders():
    folders = [Path_DnScaleYuv, Path_Bitstreams, Path_DecodedYuv, Path_QualityLog,
               Path_TestLog, Path_CfgFiles]
    if not KeepUpscaledOutput:
        folders += [Path_UpScaleYuv, Path_DecUpScaleYuv]

    for folder in folders:
        Cleanfolder(folder)

def Run_ConvexHull_Test(content, dnScalAlgo, upScalAlgo):
    Utils.Logger.info("%s %s start running xcode tests with %s" %
                      (EncodeMethod, CodecName, os.path.basename(content)))
    cls, width, height, fr, bitdepth, fmt, totalnum = GetVideoInfo(content, Clips)
    if totalnum < FrameNum:
        Utils.Logger.error("content %s includes total %d frames < specified % frames!"
                           % (content, totalnum, FrameNum))
        return

    DnScaledRes = [(int(width / ratio), int(height / ratio)) for ratio in DnScaleRatio]
    for i in range(len(DnScaledRes)):
        if SaveMemory:
            CleanIntermediateFiles()
        DnScaledW = DnScaledRes[i][0]
        DnScaledH = DnScaledRes[i][1]
        #downscaling if the downscaled file does not exist
        dnscalyuv = GetDownScaledOutFile(content, width, height, DnScaledW,
                                         DnScaledH, Path_DnScaleYuv, dnScalAlgo)
        if not os.path.isfile(dnscalyuv):
            dnscalyuv = DownScaling(content, FrameNum, width, height, DnScaledW,
                                    DnScaledH, Path_DnScaleYuv, dnScalAlgo)

        for QP in QPs:
            Utils.Logger.info("start transcode and upscale for %dx%d and QP %d"
                              % (DnScaledW, DnScaledH, QP))
            #transcode and upscaling
            reconyuv = Run_EncDec_Upscale(EncodeMethod, CodecName, EncodePreset,
                                          dnscalyuv, QP, FrameNum, fr, DnScaledW,
                                          DnScaledH, width, height, Path_Bitstreams,
                                          Path_DecodedYuv, Path_DecUpScaleYuv,
                                          upScalAlgo)
            #calcualte quality distortion
            Utils.Logger.info("start quality metric calculation for %dx%d and QP %d"
                              % (DnScaledW, DnScaledH, QP))
            CalculateQualityMetric(content, FrameNum, reconyuv, width, height,
                                   Path_QualityLog, Path_CfgFiles)

    if SaveMemory:
        Cleanfolder(Path_DnScaleYuv)

    Utils.Logger.info("finish running encode test.")

def SaveConvexHullResultsToExcel(content, dnScAlgos, upScAlgos, sum_wb,
                                 sum_start_row):
    Utils.Logger.info("start saving RD results to excel file.......")
    if not os.path.exists(Path_RDResults):
        os.makedirs(Path_RDResults)
    excFile = GetRDResultExcelFile(content)
    wb = xlsxwriter.Workbook(excFile)
    shts = []
    for i in range(len(dnScAlgos)):
        shtname = dnScAlgos[i] + '--' + upScAlgos[i]
        shts.append(wb.add_worksheet(shtname))

    cls, width, height, fr, bitdepth, fmt, totalnum = GetVideoInfo(content, Clips)
    DnScaledRes = [(int(width / ratio), int(height / ratio)) for ratio in DnScaleRatio]
    contentname = GetShortContentName(content)
    for sht, indx in zip(shts, list(range(len(dnScAlgos)))):
        # write QP
        sht.write(1, 0, "QP")
        sht.write_column(CvxH_WtRows[0], 0, QPs)
        shtname = sht.get_name()
        sum_sht = sum_wb.get_worksheet_by_name(shtname)

        charts = [];  y_mins = {}; y_maxs = {}; RDPoints = {}
        for qty, x in zip(QualityList, range(len(QualityList))):
            chart_title = 'RD Curves - %s with %s' % (contentname, shtname)
            xaxis_name = 'Bitrate - Kbps'
            chart = CreateChart_Scatter(wb, chart_title, xaxis_name, qty)
            charts.append(chart)
            y_mins[x] = []; y_maxs[x] = []; RDPoints[x] = []

        # write RD data
        for col, i in zip(CvxH_WtCols, range(len(DnScaledRes))):
            DnScaledW = DnScaledRes[i][0]
            DnScaledH = DnScaledRes[i][1]
            sht.write(0, col, "resolution=%dx%d" % (DnScaledW, DnScaledH))
            sht.write(1, col, "Bitrate(kbps)")
            sht.write_row(1, col + 1, QualityList)

            bitratesKbps = []; qualities = []
            for qp in QPs:
                bs, reconyuv = GetBsReconFileName(EncodeMethod, CodecName,
                                                  EncodePreset, content, width,
                                                  height, DnScaledW, DnScaledH,
                                                  dnScAlgos[indx], upScAlgos[indx],
                                                  qp, Path_Bitstreams)
                bitrate = (os.path.getsize(bs) * 8 * fr / FrameNum) / 1000.0
                bitratesKbps.append(bitrate)
                quality = GatherQualityMetrics(reconyuv, Path_QualityLog)
                qualities.append(quality)

            sht.write_column(CvxH_WtRows[0], col, bitratesKbps)
            for qs, row in zip(qualities, CvxH_WtRows):
                sht.write_row(row, col + 1, qs)

            seriname = "resolution %dx%d" % (DnScaledW, DnScaledH)
            for x in range(len(QualityList)):
                # add RD curves of current resolution to each quality chart
                AddSeriesToChart_Scatter(shtname, CvxH_WtRows, col + 1 + x, col,
                                         charts[x], seriname, LineColors[i])
                # get min and max of y-axis
                qs = [row[x] for row in qualities]
                y_mins[x].append(min(qs))
                y_maxs[x].append(max(qs))
                # get RD points - (bitrate, quality) for each quality metrics
                rdpnts = [(brt, qty) for brt, qty in zip(bitratesKbps, qs)]
                RDPoints[x] = RDPoints[x] + rdpnts

        # add convexhull curve to charts
        startrow = CvxH_WtRows[-1] + 2; startcol = 0

        sum_startrow = sum_start_row[shtname]
        sum_sht.write(sum_startrow, 0, cls)
        sum_sht.write(sum_startrow, 1, contentname)
        endrow, sum_end_row = AddConvexHullCurveToCharts(sht, startrow, startcol,
                                                         charts, RDPoints,
                                                         DnScaledRes, sum_sht,
                                                         sum_startrow)
        sum_start_row[shtname] = sum_end_row + 1

        #update RD chart with approprate y axis range
        for qty, x in zip(QualityList, range(len(QualityList))):
            ymin = min(y_mins[x])
            ymax = max(y_maxs[x])
            margin = 0.1  # add 10% on min and max value for y_axis range
            num_precsn = 5 if 'MS-SSIM' in qty else 3
            UpdateChart(charts[x], ymin, ymax, margin, qty, num_precsn)

        startrow = endrow + 2; startcol = 1
        InsertChartsToSheet(sht, startrow, startcol, charts)

    wb.close()
    Utils.Logger.info("finish export convex hull results to excel file.")

def ParseArguments(raw_args):
    parser = argparse.ArgumentParser(prog='ConvexHullTest.py',
                                     usage='%(prog)s [options]',
                                     description='')
    parser.add_argument('-f', '--function', dest='Function', type=str,
                        required=True, metavar='',
                        choices=["clean", "scaling", "sumscaling", "encode",
                                 "convexhull", "summary"],
                        help="function to run: clean, scaling, sumscaling, encode,"
                             " convexhull, summary")
    parser.add_argument('-k', "--KeepUpscaleOutput", dest='KeepUpscaledOutput',
                        type=bool, default=False, metavar='',
                        help="in function clean, if keep upscaled yuv files. It"
                             " is false by default")
    parser.add_argument('-s', "--SaveMemory", dest='SaveMemory', type=bool,
                        default=False, metavar='',
                        help="save memory mode will delete most files in"
                             " intermediate steps and keeps only necessary "
                             "ones for RD calculation. It is false by default")
    parser.add_argument('-l', "--LoggingLevel", dest='LogLevel', type=int,
                        default=3, choices=range(len(LogLevels)), metavar='',
                        help="logging level: 0:No Logging, 1: Critical, 2: Error,"
                             " 3: Warning, 4: Info, 5: Debug")
    parser.add_argument('-c', "--CodecName", dest='CodecName', type=str,
                        choices=CodecNames, metavar='',
                        help="CodecName: av1 or hevc")
    parser.add_argument('-m', "--EncodeMethod", dest='EncodeMethod', type=str,
                        choices=EncodeMethods, metavar='',
                        help="EncodeMethod: ffmpeg, aom, svt")
    parser.add_argument('-p', "--EncodePreset", dest='EncodePreset', type=str,
                        metavar='', help="EncodePreset: medium, slow, fast, etc"
                                         " for ffmpeg, 0,1,2... for aom and svt")
    if len(raw_args) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args(raw_args[1:])

    global Function, KeepUpscaledOutput, SaveMemory, LogLevel, CodecName,\
        EncodeMethod, EncodePreset
    Function = args.Function
    KeepUpscaledOutput = args.KeepUpscaledOutput
    SaveMemory = args.SaveMemory
    LogLevel = args.LogLevel
    CodecName = args.CodecName
    EncodeMethod = args.EncodeMethod
    EncodePreset = args.EncodePreset


######################################
# main
######################################
if __name__ == "__main__":
    #sys.argv = ["","-f","convexhull","-c","hevc","-m","ffmpeg","-p","medium"]
    ParseArguments(sys.argv)

    # preparation for executing functions
    setupWorkFolderStructure()
    if Function != 'clean':
        SetupLogging(LogLevel, LogCmdOnly, LoggerName, Path_TestLog)
        Contents = GetContents(ContentPath, Clips)

    # execute functions
    if Function == 'clean':
        CleanUp_workfolders()
    elif Function == 'scaling':
        for content in Contents:
            for dnScaleAlgo, upScaleAlgo in zip(DnScalingAlgos, UpScalingAlgos):
                Run_Scaling_Test(content, dnScaleAlgo, upScaleAlgo,
                                 Path_DnScaleYuv, Path_UpScaleYuv, Path_QualityLog,
                                 Path_CfgFiles, SaveMemory, KeepUpscaledOutput)
    elif Function == 'sumscaling':
        SaveScalingResultsToExcel(DnScalingAlgos, UpScalingAlgos, Path_QualityLog)
    elif Function == 'encode':
        for content in Contents:
            for dnScalAlgo, upScalAlgo in zip(DnScalingAlgos, UpScalingAlgos):
                Run_ConvexHull_Test(content, dnScalAlgo, upScalAlgo)
    elif Function == 'convexhull':
        sum_wb, sum_start_row = GenerateSummaryConvexHullExcelFile(EncodeMethod,
                                                                   CodecName,
                                                                   EncodePreset,
                                                                   SummaryOutPath,
                                                                   DnScalingAlgos,
                                                                   UpScalingAlgos)
        for content in Contents:
            SaveConvexHullResultsToExcel(content, DnScalingAlgos, UpScalingAlgos,
                                         sum_wb, sum_start_row)
        sum_wb.close()
    elif Function == 'summary':
        smfile = GenerateSummaryExcelFile(EncodeMethod, CodecName, EncodePreset,
                                          SummaryOutPath, Path_RDResults,
                                          ContentPath, Clips)
        Utils.Logger.info("summary file generated: %s" % smfile)
    else:
        Utils.Logger.error("invalid parameter value of Function")
