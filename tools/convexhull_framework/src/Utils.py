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
import xlrd
import subprocess
import time
import logging
from Config import LogLevels, Have_Class_Subfolder

def Cleanfolder(folder):
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            file = os.path.join(folder, f)
            if os.path.isfile(file):
                os.remove(file)

def CreateNewSubfolder(parent, name):
    if name == '' or name == None:
        return None
    folder = os.path.join(parent, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def GetShortContentName(content, isshort=True):
    basename = os.path.splitext(os.path.basename(content))[0]
    if isshort:
        item = re.findall(r"([a-zA-Z0-9]+)_", basename)
        if len(item) == 0:
            name = basename
        else:
            name = item[0]
    else:
        name = basename
    return name

def GetContents(contentpath, clips):
    contents = []
    for key, val in clips.items():
        folder = contentpath
        if Have_Class_Subfolder:
            cls = val[0]
            folder = os.path.join(contentpath, cls)

        file = os.path.join(folder, key) + ".yuv"
        if os.path.isfile(file):
            contents.append(file)

    return contents

def GetVideoInfo(content, Clips):
    basename = GetShortContentName(content, False)
    cls = Clips[basename][0]
    width = Clips[basename][1]
    height = Clips[basename][2]
    fr = Clips[basename][3]
    bitdepth = Clips[basename][4]
    fmt = Clips[basename][5]

    #default for 8 bit 420
    RatioForFrameSize = 3/2
    if fmt == 'yuv422p':
        RatioForFrameSize = 2
    elif fmt == 'yuv444p':
        RatioForFrameSize = 3
    if bitdepth > 8:
        RatioForFrameSize *= 2

    totalnum = os.path.getsize(content) / (width * height * RatioForFrameSize)

    return cls, width, height, fr, bitdepth, fmt, totalnum

def GetContentDict(contentpath, clips):
    dict = {}

    if Have_Class_Subfolder:
        for key, val in clips.items():
            cls = val[0]
            folder = os.path.join(contentpath, cls)
            file = os.path.join(folder, key) + ".yuv"
            if os.path.isfile(file):
                if cls in dict:
                    if file not in dict[cls]:
                        dict[cls].append(file)
                else:
                    dict[cls] = [file]
    else:
        # this is for case no subfolder/class. As * is forbidden for folder name,
        # no folder will be same as this
        cls = "*All"
        dict[cls] = GetContents(contentpath, clips)

    return dict

def CalcRowsClassAndContentDict(rowstart, content_path, clips, times=1):
    contentsdict = GetContentDict(content_path, clips)

    ofstc = rowstart
    rows_class = []
    for cls, contents in contentsdict.items():
        rows_class.append(ofstc)
        ofstc = ofstc + len(contents) * times

    return contentsdict, rows_class

def SweepScalingAlgosInOneResultFile(infile_path):
    dnscls = []
    upscls = []
    resultfiles = os.listdir(infile_path)
    # here assume all result files includes same combinations of dn and up
    # scaling algos
    file = os.path.join(infile_path, resultfiles[0])
    if os.path.isfile(file):
        rdwb = xlrd.open_workbook(os.path.join(infile_path, resultfiles[0]))
    else:
        return dnscls, upscls
    if rdwb is not None:
        shtnms = rdwb.sheet_names()
        for shtname in shtnms:
            item = re.findall(r"(.+)\-\-(.+)", shtname)
            dnsl = item[0][0]
            upsl = item[0][1]
            dnscls.append(dnsl)
            upscls.append(upsl)

    return dnscls, upscls

def CreateChart_Scatter(wb, title, xaxis_name, yaxis_name):
    chart = wb.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
    chart.set_title({'name': title, 'name_font': {'color': 'white'}})
    chart.set_x_axis({'name': xaxis_name,
                      'major_gridlines': {'visible': True, 'line': {'width': 0.25}},
                      'name_font': {'color': 'white'},
                      'num_font': {'color': 'white', 'transparency': 80},
                      'label_position' : 'low'
                      })
    chart.set_y_axis({'name': yaxis_name, 'name_font': {'color': 'white'},
                      'num_font': {'color': 'white'}})
    chart.set_style(12)
    chart.set_size({'x_scale': 1.5, 'y_scale': 1.5})
    chart.set_chartarea({"fill": {'color': '#505050'}})
    chart.set_plotarea({"fill": {'color': '#505050'}})
    chart.set_legend({'position': 'bottom', 'font': {'color': 'white'}})
    return chart

def CreateChart_Line(wb, titlename, yaxis_name):
    chart = wb.add_chart({'type': 'line', 'name_font': {'size': 10.5}})
    chart.set_title({'name': titlename})
    chart.set_x_axis({'text_axis': True})
    chart.set_y_axis({'name': yaxis_name, 'name_font': {'size': 11}})
    chart.set_size({'x_scale': 1.4, 'y_scale': 1.5})
    chart.set_legend({'position': 'right', 'font': {'size': 10.5}})
    chart.set_high_low_lines(
        {'line': {'color': 'black', 'size': 2}}
    )
    return chart

def AddSeriesToChart_Scatter(shtname, rows, coly, colx, chart, seriname,
                             linecolor):
    yvalues = [shtname, rows[0], coly, rows[-1], coly]
    xvalues = [shtname, rows[0], colx, rows[-1], colx]

    chart.add_series({
        'name': seriname,
        'categories': xvalues,
        'values': yvalues,
        'line': {'color': linecolor, 'width': 1.5},
        'marker': {'type': 'circle', 'size': 5,
                   'border': {'color': linecolor, 'size': 0.75},
                   'fill': {'color': linecolor}},
    })

def AddSeriesToChart_Scatter_Rows(shtname, cols, rowy, rowx, chart, seriname,
                                  linecolor):
    yvalues = [shtname, rowy, cols[0], rowy, cols[-1]]
    xvalues = [shtname, rowx, cols[0], rowx, cols[-1]]

    chart.add_series({
        'name': seriname,
        'categories': xvalues,
        'values': yvalues,
        'line': {'color': linecolor, 'width': 1.0, 'dash_type': 'dash_dot'},
        'marker': {'type': 'square', 'size': 5,
                   'border': {'color': 'white', 'size': 0.75}}
    })

def AddSeriesToChart_Line(shtname, rows, coly, colx, chart, seriname, shape,
                          ssize, linecolor):
    yvalues = [shtname, rows[0], coly, rows[-1], coly]
    xvalues = [shtname, rows[0], colx, rows[-1], colx]
    chart.add_series({
        'name': seriname,
        'categories': xvalues,
        'values': yvalues,
        'line': {'none': True},
        'marker': {'type': shape,
                   'size': ssize,
                   'border': {'color': linecolor, 'size': 2},
                   'fill': {'color': linecolor}},
    })

def UpdateChart(chart, ymin, ymax, margin, yaxis_name, precsn):
    interval = ymax - ymin
    finalmax = ymax + interval * margin
    finalmin = ymin - interval * margin
    floatprecn = "{:.%df}" % precsn
    finalmin = float(floatprecn.format(finalmin))
    finalmax = float(floatprecn.format(finalmax))
    chart.set_y_axis({'name': yaxis_name,
                      'name_font': {'color': 'white'},
                      'num_font': {'color': 'white'},
                      'min': finalmin, 'max': finalmax})

def InsertChartsToSheet(sht, startrow, startcol, charts):
    height = 22
    width = 12
    num = len(charts)
    row = startrow
    for i in range(1, num, 2):
        sht.insert_chart(row, startcol, charts[i - 1])
        sht.insert_chart(row, startcol + width, charts[i])
        row = row + height

def ExecuteCmd(cmd, LogCmdOnly):
    CmdLogger.write(cmd + "\n")
    ret = 0
    if not LogCmdOnly:
        ret = subprocess.call(cmd, shell=True)
    return ret

def SetupLogging(level, logcmdonly, name, path):
    global Logger
    Logger = logging.getLogger(name)

    if logcmdonly or level != 0:
        global CmdLogger
        logfilename = os.path.join(path, 'ConvexHullTestCmd_%s.log'
                                   % time.strftime("%Y%m%d-%H%M%S"))
        CmdLogger = open(logfilename, 'w')

    if level != 0:
        logfilename = os.path.join(path, 'ConvexHullTest_%s.log'
                                   % time.strftime("%Y%m%d-%H%M%S"))
        hdlr = logging.FileHandler(logfilename)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        hdlr.setFormatter(formatter)
        Logger.addHandler(hdlr)
        if level in range(len(LogLevels)):
            # valid level input parameter
            lvl = LogLevels[level]
            levelname = logging.getLevelName(lvl)
        else:
            # if not valid, default set to 'INFO'
            levelname = logging.getLevelName('INFO')
        Logger.setLevel(levelname)
