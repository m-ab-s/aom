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
import xlsxwriter
import xlrd
from Config import QPs, DnScaleRatio, QualityList, VbaBinFile, CvxH_WtRows,\
    CvxH_WtLastCol, LoggerName
from Utils import GetShortContentName, CalcRowsClassAndContentDict,\
    SweepScalingAlgosInOneResultFile
import logging

subloggername = "PostAnalysisSummary"
loggername = LoggerName + '.' + '%s' % subloggername
logger = logging.getLogger(loggername)

# give all paths including convex hull result file (only one file for each
# content) to generate summary file for all contents in Input path
# assume all content's result has same test settings

################################################################################
### Helper Functions ###########################################################
def GetSummaryFileName(encMethod, codecName, preset, path):
    name = 'ConvexHullSummary_ScaleAlgosNum_%d_%s_%s_%s.xlsm'\
           % (len(DnScaleRatio), encMethod, codecName, preset)
    return os.path.join(path, name)

def GetConvexHullRDFileName(encMethod, codecName, preset, path):
    name = 'ConvexHullRD_ScaleAlgosNum_%d_%s_%s_%s.xlsx'\
           % (len(DnScaleRatio), encMethod, codecName, preset)
    return os.path.join(path, name)

def CopyResultDataToSummaryFile_Onesheet(sht, wt_cols, contentsdict, rows_class,
                                         infile_path):
    rdrows = CvxH_WtRows
    rd_endcol = CvxH_WtLastCol

    shtname = sht.get_name()
    sht.write(1, 0, 'Content Class')
    sht.write(1, 1, 'Content Name')
    sht.write(1, 2, 'QP')
    for residx, col in zip(range(len(DnScaleRatio)), wt_cols):
        sht.write(0, col, 'Scaling Ratio = %.2f' % (DnScaleRatio[residx]))
        sht.write(1, col, 'Bitrate(kbps)')
        qtynames = ['%s' % qty for qty in QualityList]
        sht.write_row(1, col + 1, qtynames)

    # copy the results data from each content's result file to corresponding
    # location in summary excel file
    resultfiles = os.listdir(infile_path)
    for (cls, contents), row_class in zip(contentsdict.items(), rows_class):
        sht.write(row_class, 0, cls)
        rows_content = [i * len(QPs) for i in range(len(contents))]
        for content, row_cont in zip(contents, rows_content):
            key = GetShortContentName(content)
            sht.write(row_class + row_cont, 1, key)
            rdwb = None
            for resfile in resultfiles:
                if key in resfile:
                    rdwb = xlrd.open_workbook(os.path.join(infile_path, resfile))
                    rdsht = rdwb.sheet_by_name(shtname)
                    for i, rdrow in zip(range(len(QPs)), rdrows):
                        data = rdsht.row_values(rdrow, 0, rd_endcol + 1)
                        sht.write_row(row_class + row_cont + i, 2, data)
                    break
            assert rdwb is not None
            if rdwb is None:
                logger.warning("not find convex hull result file for content:%s"
                               % content)

def CalBDRate_OneSheet(sht,  cols, contentsdict, rows_class, cols_bdmtrs, cellformat):
    row_refst = 0
    bdstep = 3
    for cols_bd, residx in zip(cols_bdmtrs, range(1, len(DnScaleRatio))):
        sht.write(0, cols_bd, 'BD-Rate %.2f vs. %.2f' % (DnScaleRatio[residx],
                                                         DnScaleRatio[0]))
        sht.write_row(1, cols_bd, QualityList)
        for (cls, contents), row_class in zip(contentsdict.items(), rows_class):
            rows_content = [i * len(QPs) for i in range(len(contents))]
            for row_cont in rows_content:
                for y in range(len(QualityList)):
                    refbr_b = xlrd.cellnameabs(row_class + row_cont + row_refst,
                                               cols[0])
                    refbr_e = xlrd.cellnameabs(row_class + row_cont + row_refst
                                               + bdstep, cols[0])
                    refq_b = xlrd.cellnameabs(row_class + row_cont + row_refst,
                                              cols[0] + 1 + y)
                    refq_e = xlrd.cellnameabs(row_class + row_cont + row_refst
                                              + bdstep, cols[0] + 1 + y)

                    testbr_b = xlrd.cellnameabs(row_class + row_cont,
                                                cols[residx])
                    testbr_e = xlrd.cellnameabs(row_class + row_cont + bdstep,
                                                cols[residx])
                    testq_b = xlrd.cellnameabs(row_class + row_cont,
                                               cols[residx] + 1 + y)
                    testq_e = xlrd.cellnameabs(row_class + row_cont + bdstep,
                                               cols[residx] + 1 + y)

                    #formula = '=-bdrate(%s:%s,%s:%s,%s:%s,%s:%s)' % (
                    #refbr_b, refbr_e, refq_b, refq_e, testbr_b, testbr_e,
                    # testq_b, testq_e)
                    formula = '=bdRateExtend(%s:%s,%s:%s,%s:%s,%s:%s)'\
                              % (refbr_b, refbr_e, refq_b, refq_e, testbr_b,
                                 testbr_e, testq_b, testq_e)
                    sht.write_formula(row_class + row_cont, cols_bd + y, formula,
                                      cellformat)

def GenerateFormula_SumRows(shtname, rows, col):
    cells = ''
    for row in rows:
        location = xlrd.cellnameabs(row, col)
        cells = cells + '\'%s\'!%s,' % (shtname, location)
    cells = cells[:-1]  # remove the last ,
    formula = '=SUM(%s)/%d' % (cells, len(rows))
    return formula

def GenerateFormula_SumRows_Weighted(rows, col, weight_rows, weight_col, num):
    cells = ''
    for row, wtrow in zip(rows, weight_rows):
        location = xlrd.cellnameabs(row, col)
        weight = xlrd.cellnameabs(wtrow, weight_col)
        cells = cells + '%s * %s,' % (location, weight)
    cells = cells[:-1]  # remove the last ,
    formula = '=SUM(%s)/%d' % (cells, num)
    return formula

def WriteBitrateQtyAverageSheet(wb, rdshts, contentsdict, rd_rows_class, rdcols):
    avg_sht = wb.add_worksheet('Average')
    avg_sht.write(2, 0, 'Content Class')
    avg_sht.write(2, 1, 'Content Number')
    avg_sht.write(2, 2, 'QP')

    colstart = 3
    cols_res = [colstart]
    step = len(QualityList) + 1 + 1  # 1 for bitrate, 1 for interval
    colres_2nd_start = colstart + step
    step = len(upScalAlgos) * (len(QualityList) + 1) + 1  # + 1 for interval
    cols_res += [step * i + colres_2nd_start for i in range(len(DnScaleRatio) - 1)]
    step = len(QualityList) + 1  # + 1 for bitrate
    cols_upscl = [step * i for i in range(len(upScalAlgos))]
    for residx, col_res in zip(range(len(DnScaleRatio)), cols_res):
        avg_sht.write(0, col_res, 'ScalingRatio = %.2f' % (DnScaleRatio[residx]))
        if residx == 0:
            avg_sht.write(1, col_res + 1, 'None')
            avg_sht.write(2, col_res, 'Bitrate(kbps)')
            avg_sht.write_row(2, col_res + 1, QualityList)
        else:
            for dnsc, upsc, col_upscl in zip(dnScalAlgos, upScalAlgos, cols_upscl):
                avg_sht.write(1, col_res + col_upscl + 1, '%s--%s' % (dnsc, upsc))
                avg_sht.write(2, col_res + col_upscl, 'Bitrate(kbps)')
                avg_sht.write_row(2, col_res + col_upscl + 1, QualityList)

    startrow = 3
    step = len(QPs)
    rows_class_avg = [startrow + step * i for i in range(len(contentsdict))]
    totalnum_content = 0
    for (cls, contents), row_class, rdclassrow in zip(contentsdict.items(),
                                                      rows_class_avg,
                                                      rd_rows_class):
        avg_sht.write(row_class, 0, cls)
        totalnum_content = totalnum_content + len(contents)
        avg_sht.write(row_class, 1, len(contents))
        avg_sht.write_column(row_class, 2, QPs)
        rows_content = [i * len(QPs) for i in range(len(contents))]

        for rdcol, col_res, residx in zip(rdcols, cols_res, range(len(DnScaleRatio))):
            for i in range(len(QPs)):
                sum_rows = [rdclassrow + row_cont + i for row_cont in rows_content]
                for col_upscl, sht in zip(cols_upscl, rdshts):
                    shtname = sht.get_name()
                    # write bitrate average formula.
                    formula = GenerateFormula_SumRows(shtname, sum_rows, rdcol)
                    avg_sht.write_formula(row_class + i, col_res + col_upscl,
                                          formula)
                    # write quality average formula
                    for j in range(len(QualityList)):
                        formula = GenerateFormula_SumRows(shtname, sum_rows,
                                                          rdcol + 1 + j)
                        avg_sht.write_formula(row_class + i,
                                              col_res + col_upscl + 1 + j,
                                              formula)

                    # for first resolution, no down and up scaling. only need
                    # one set of bitrate/quality data
                    if residx == 0:
                        break

    # write total average
    last_class_row = rows_class_avg[-1] + len(QPs) + 1  # 1 for 1 row of interval
    avg_sht.write(last_class_row, 0, 'Total')
    avg_sht.write(last_class_row, 1, totalnum_content)
    avg_sht.write_column(last_class_row, 2, QPs)
    weight_rows = [row_class for row_class in rows_class_avg]
    for col_res, residx in zip(cols_res, range(len(DnScaleRatio))):
        for i in range(len(QPs)):
            sum_rows = [row_class + i for row_class in rows_class_avg]
            for col_upscl in cols_upscl:
                # bitrate average
                formula = GenerateFormula_SumRows_Weighted(sum_rows,
                                                           col_res + col_upscl,
                                                           weight_rows, 1,
                                                           totalnum_content)
                avg_sht.write_formula(last_class_row + i, col_res + col_upscl,
                                      formula)
                # quality average
                for j in range(len(QualityList)):
                    formula = GenerateFormula_SumRows_Weighted(sum_rows,
                                                               col_res +
                                                               col_upscl + 1 + j,
                                                               weight_rows, 1,
                                                               totalnum_content)
                    avg_sht.write_formula(last_class_row + i,
                                          col_res + col_upscl + 1 + j, formula)
                # for first resolution, no down and up scaling. only need one
                # set of bitrate/quality data
                if residx == 0:
                    break

def WriteBDRateAverageSheet(wb, rdshts, contentsdict, rd_rows_class,
                            rd_cols_bdmtrs, cellformat):
    # write bdrate average sheet
    bdavg_sht = wb.add_worksheet('Average_BDRate')
    bdavg_sht.write(2, 0, 'Content Class')
    bdavg_sht.write(2, 1, 'Content Number')

    startcol = 2
    startrow = 3
    colintval_scalalgo = 1
    colintval_dnscalres = 1

    step_upscl = len(QualityList) + colintval_scalalgo
    cols_upscl_bd = [step_upscl * i for i in range(len(upScalAlgos))]
    step_res = len(upScalAlgos) * step_upscl + colintval_dnscalres
    cols_res_bd = [step_res * i + startcol for i in range(len(DnScaleRatio) - 1)]
    rows_class_rdavg = [startrow + i for i in range(len(contentsdict))]

    for residx, col_res_bd in zip(range(1, len(DnScaleRatio)), cols_res_bd):
        bdavg_sht.write(0, col_res_bd, 'BD-Rate %.2f vs. %.2f'
                        % (DnScaleRatio[residx], DnScaleRatio[0]))
        for dnsc, upsc, col_upscl_bd in zip(dnScalAlgos, upScalAlgos, cols_upscl_bd):
            bdavg_sht.write(1, col_res_bd + col_upscl_bd, '%s--%s' % (dnsc, upsc))
            bdavg_sht.write_row(2, col_res_bd + col_upscl_bd, QualityList)

    totalnum_content = 0
    for (cls, contents), row_class, rdclassrow in zip(contentsdict.items(),
                                                      rows_class_rdavg,
                                                      rd_rows_class):
        bdavg_sht.write(row_class, 0, cls)
        totalnum_content = totalnum_content + len(contents)
        bdavg_sht.write(row_class, 1, len(contents))
        rows_content = [i * len(QPs) for i in range(len(contents))]
        sum_rows = [rdclassrow + row_cont for row_cont in rows_content]
        for rdcol, col_res in zip(rd_cols_bdmtrs, cols_res_bd):
            # write average bd rate
            for col_upscl, sht in zip(cols_upscl_bd, rdshts):
                shtname = sht.get_name()
                for j in range(len(QualityList)):
                    formula = GenerateFormula_SumRows(shtname, sum_rows, rdcol + j)
                    bdavg_sht.write_formula(row_class, col_res + col_upscl + j,
                                            formula, cellformat)

    # write total average
    last_row = rows_class_rdavg[-1] + 1
    bdavg_sht.write(last_row, 0, 'Total')
    bdavg_sht.write(last_row, 1, totalnum_content)
    sum_rows = [row_class for row_class in rows_class_rdavg]
    for col_res in cols_res_bd:
        for col_upscl in cols_upscl_bd:
            for j in range(len(QualityList)):
                formula = GenerateFormula_SumRows_Weighted(sum_rows,
                                                           col_res + col_upscl + j,
                                                           sum_rows, 1,
                                                           totalnum_content)
                bdavg_sht.write_formula(last_row, col_res + col_upscl + j,
                                        formula, cellformat)

#######################################################################
#######################################################################
# GenerateSummaryExcelFile is to
# 1. summarize all contents convexhull results into one file
# 2. calculate average of bitrate and quality metrics for each content class
# 3. calculate BD rate across different scaling ratios for all scaling
#    algorithms in convex hull test
# 4. calcualte average BD rate for each content class
# Arguments description:
# content_paths is where test contents located, which used for generating convex
#               hull results.
# infile_path   is where convex hull results excel files located.
# classes       is the content classes and  here it requires contents are located
#               in corresponding subfolder named with its belonged class
# summary_outpath  is the folder where output summary file will be
# note: all results files under infile_path should have exactly same test items
# before running this summary script
def GenerateSummaryExcelFile(encMethod, codecName, preset, summary_outpath,
                             infile_path, content_path, clips):
    global dnScalAlgos, upScalAlgos
    # find all scaling algos tested in results file, expect they are the same
    # for every content
    dnScalAlgos, upScalAlgos = SweepScalingAlgosInOneResultFile(infile_path)

    if not os.path.exists(summary_outpath):
        os.makedirs(summary_outpath)
    smfile = GetSummaryFileName(encMethod, codecName, preset, summary_outpath)
    wb = xlsxwriter.Workbook(smfile)

    # shts is for all scaling algorithms' convex hull test results
    shts = []
    for dnsc, upsc in zip(dnScalAlgos, upScalAlgos):
        shtname = dnsc + '--' + upsc
        sht = wb.add_worksheet(shtname)
        shts.append(sht)

    # below variables define summary file data layout format.
    # if to change them, modify CopyResultsDataToSummaryFile_Onesheet() and
    # CalcRowsCategAndContentDict() accordingly
    colstart = 3
    colInterval = 2
    rowstart = 2

    # cols is column number of results files
    step = colInterval + 1 + len(QualityList)  # 1 is for bitrate
    sum_wtcols = [step * i + colstart for i in range(len(DnScaleRatio))]
    # to generate rows number of starting of each class: rows_class
    contentsdict, rows_class = CalcRowsClassAndContentDict(rowstart,
                                                           content_path,
                                                           clips, len(QPs))

    wb.add_vba_project(VbaBinFile)
    cellformat = wb.add_format()
    cellformat.set_num_format('0.00%')
    #cols_bdmtrs is the column number to write the bdrate data
    step = len(QualityList) + 1
    start_col_bd = sum_wtcols[-1] + step + 1
    cols_bdmtrs = [start_col_bd + i * step for i in range(len(DnScaleRatio) - 1)]
    # -1 because first resolution is used as reference

    for sht in shts:
        CopyResultDataToSummaryFile_Onesheet(sht, sum_wtcols, contentsdict,
                                             rows_class, infile_path)
        # calculate bd rate in each scaling sheet
        CalBDRate_OneSheet(sht, sum_wtcols, contentsdict, rows_class,
                           cols_bdmtrs, cellformat)

    # calculate average bitrate and quality metrics for each category and
    # write to "average" sheet
    WriteBitrateQtyAverageSheet(wb, shts, contentsdict, rows_class, sum_wtcols)

    # calculate average bd metrics and write to a new sheet
    WriteBDRateAverageSheet(wb, shts, contentsdict, rows_class, cols_bdmtrs,
                            cellformat)

    wb.close()
    return smfile

def GenerateSummaryConvexHullExcelFile(encMethod, codecName, preset,
                                       summary_outpath, DnScalingAlgos,
                                       UpScalingAlgos):
    if not os.path.exists(summary_outpath):
        os.makedirs(summary_outpath)
    smfile = GetConvexHullRDFileName(encMethod, codecName, preset,
                                     summary_outpath)
    sum_wb = xlsxwriter.Workbook(smfile)

    # shts is for all scaling algorithms' convex hull test results
    sum_start_row = {}
    #write the header in each sheet
    for dnsc, upsc in zip(DnScalingAlgos, UpScalingAlgos):
        shtname = dnsc + '--' + upsc
        sht = sum_wb.add_worksheet(shtname)
        sht.write(0, 0, 'Content Class')
        sht.write(0, 1, 'Content Name')
        sht.write(0, 2, 'Num RD Points')
        col = 3
        for qty in QualityList:
            sht.write(0, col, 'Bitrate(kbps)')
            sht.write(0, col + 1, qty)
            col += 2
        sum_start_row[shtname] = 1
    return sum_wb, sum_start_row
