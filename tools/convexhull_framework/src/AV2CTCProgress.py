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
__author__ = "maggie.sun@intel.com, ryanlei@fb.com"

import re
import openpyxl
import shutil
import Config
from Config import QPs, DnScaleRatio, CTC_ASXLSTemplate, CTC_RegularXLSTemplate
import Utils
from Utils import ParseCSVFile, plot_rd_curve, Interpolate_Bilinear, convex_hull
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

csv_files = {
    "v1.0.0":
    {
        "AI":  "D:\\AV2-CTC-v1.0.0-Final\\analysis\\rdresult\\RDResults_aom_av2_AI_Preset_0.csv",
        "LD":  "D:\\AV2-CTC-v1.0.0-Final\\analysis\\rdresult\\RDResults_aom_av2_LD_Preset_0.csv",
        "RA":  "D:\\AV2-CTC-v1.0.0-Final\\analysis\\rdresult\\RDResults_aom_av2_RA_Preset_0.csv",
        "Still":  "D:\\AV2-CTC-v1.0.0-Final\\analysis\\rdresult\\RDResults_aom_av2_STILL_Preset_0.csv",
        "AS":  "D:\\AV2-CTC-v1.0.0-Final\\analysis\\rdresult\\RDResults_aom_av2_AS_Preset_0.csv",
    },
    "v1.0.1":
    {
        "AI":  "D:\\AV2-CTC-v1.0.1\\analysis\\rdresult\\RDResults_aom_av2_AI_Preset_0.csv",
        "LD":  "D:\\AV2-CTC-v1.0.1\\analysis\\rdresult\\RDResults_aom_av2_LD_Preset_0.csv",
        "RA":  "D:\\AV2-CTC-v1.0.1\\analysis\\rdresult\\RDResults_aom_av2_RA_Preset_0.csv",
        "Still":  "D:\\AV2-CTC-v1.0.1\\analysis\\rdresult\\RDResults_aom_av2_STILL_Preset_0.csv",
        "AS":  "D:\\AV2-CTC-v1.0.1\\analysis\\rdresult\\RDResults_aom_av2_AS_Preset_0.csv",
    },
}

start_row = {
    "AI": 2,
    "AS": 2,
    "RA": 2,
    "Still": 2,
    "LD": 50,
}

formats = {
    "v1.0.0": ['r', '-', 'o'],
    "v1.0.1": ['g', '-', '*'],
}

AS_formats = {
    "3840x2160": ['r', '-.', 'o'],
    "2560x1440": ['g', '-.', '*'],
    "1920x1080": ['b', '-.', '^'],
    "1280x720":  ['y', '-.', '+'],
    "960x540":   ['c', '-.', 'x'],
    "640x360":   ['k', '-.', '<'],
}

anchor = "v1.0.0"
rd_curve_pdf = "rdcurve.pdf"

def WriteSheet(csv_file, sht, start_row):
    csv = open(csv_file, 'rt')
    row = start_row
    for line in csv:
        if not line.startswith('TestCfg'):
            words = re.split(',', line.strip())
            col = 1
            for word in words:
                mycell = sht.cell(row=row, column=col)
                if col >= 12 and word != "":
                    mycell.value = float(word)
                else:
                    mycell.value = word
                col += 1
            row += 1
    csv.close()


def FillXlsFile():
    for tag in csv_files.keys():
        if tag == anchor:
            continue
        else:
            for cfg in csv_files[anchor].keys():
                anchor_sht_name = "Anchor-%s" % cfg
                test_sht_name = "Test-%s" % cfg
                if cfg == "AS":
                    xls_template = CTC_ASXLSTemplate
                    xls_file = "CTC_AS_%s-%s.xlsm" % (anchor, tag)
                    shutil.copyfile(xls_template, xls_file)
                    anchor_sht_name = "Anchor"
                    test_sht_name = "Test"
                elif cfg == "AI":
                    xls_template = CTC_RegularXLSTemplate
                    xls_file = "CTC_Regular_%s-%s.xlsm" % (anchor, tag)
                    shutil.copyfile(xls_template, xls_file)

                wb = openpyxl.load_workbook(filename=xls_file, read_only=False, keep_vba=True)
                anchor_sht = wb[anchor_sht_name]
                anchor_csv = csv_files[anchor][cfg]
                WriteSheet(anchor_csv, anchor_sht, start_row[cfg])

                test_sht = wb[test_sht_name]
                test_csv = csv_files[tag][cfg]
                WriteSheet(test_csv, test_sht, start_row[cfg])

                wb.save(xls_file)

def DrawRDCurve(records, anchor, pdf):
    with PdfPages(pdf) as export_pdf:
        for cfg in records[anchor].keys():
            videos = records[anchor][cfg].keys()
            for video in videos:
                if cfg == "AS":
                    DnScaledRes = [(int(3840/ratio), int(2160/ratio)) for ratio in DnScaleRatio]
                    Int_RDPoints = {}
                    # draw individual rd curves
                    for tag in records.keys():
                        Int_RDPoints[tag] = []
                        record = records[tag][cfg][video]
                        plt.figure(figsize=(15, 10))
                        plt.suptitle("%s : %s: %s" % (cfg, video, tag))

                        for (w, h) in DnScaledRes:
                            br = []; apsnr = []; rdpnts = []; res = "%dx%d"%(w,h)
                            for qp in QPs["AS"]:
                                key = "%dx%d_%s"%(w, h, qp)
                                br.append(record[key].bitrate)
                                apsnr.append(record[key].overall_apsnr)
                            rdpnts = [(brt, qty) for brt, qty in zip(br, apsnr)]
                            int_rdpnts = Interpolate_Bilinear(rdpnts, QPs['AS'][:], True)
                            Int_RDPoints[tag] += int_rdpnts
                            plot_rd_curve(br, apsnr, "overall_apsnr", res, "bitrate(Kbps)",
                                          AS_formats[res][0], AS_formats[res][1], AS_formats[res][2])
                        plt.legend()
                        plt.grid(True)
                        export_pdf.savefig()
                        plt.close()

                    #draw convex hull
                    plt.figure(figsize=(15, 10))
                    plt.suptitle("%s : %s: convex hull" % (cfg, video))
                    for tag in records.keys():
                        lower, upper = convex_hull(Int_RDPoints[tag])
                        br    = [h[0] for h in upper]
                        apsnr = [h[1] for h in upper]
                        plot_rd_curve(br, apsnr, "overall_apsnr", tag, "bitrate(Kbps)",
                                      formats[tag][0], formats[tag][1], formats[tag][2])
                    plt.legend()
                    plt.grid(True)
                    export_pdf.savefig()
                    plt.close()
                else:
                    # bit rate to quality
                    plt.figure(figsize=(15, 10))
                    plt.suptitle("%s : %s" % (cfg, video))
                    for tag in records.keys():
                        record = records[tag][cfg][video]
                        br    = [record[key].bitrate for key in record.keys()]
                        apsnr = [record[key].overall_apsnr for key in record.keys()]
                        plot_rd_curve(br, apsnr, "overall_apsnr", tag, "bitrate(Kbps)",
                                      formats[tag][0], formats[tag][1], formats[tag][2])
                    plt.legend()
                    plt.grid(True)
                    export_pdf.savefig()
                    plt.close()


######################################
# main
######################################
if __name__ == "__main__":
    records = {}
    for tag in csv_files.keys():
        records[tag] = {}
        for test_cfg in csv_files[tag].keys():
            records[tag][test_cfg] = ParseCSVFile(csv_files[tag][test_cfg])

    FillXlsFile()

    DrawRDCurve(records, anchor, rd_curve_pdf)