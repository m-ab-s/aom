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

"""
Python file for definition of AV2 CTC testing clips/sets
"""

CTC_TEST_SET = ["a1_4k", "a2_2k", "a3_720p", "a4_360p", "a5_270p", "b1_syn", "hdr1_4k", "hdr2_2k"]
AS_TEST_SET = ["a1_4k"]
Y4M_CLIPs = {
"a1_4k"         : ["BoxingPractice_3840x2160_5994fps_10bit_420.y4m",
                   "Crosswalk_3840x2160_5994fps_10bit_420.y4m",
                   "FoodMarket2_3840x2160_5994fps_10bit_420.y4m",
                   "Neon1224_3840x2160_2997fps.y4m",
                   "NocturneDance_3840x2160p_10bit_60fps.y4m",
                   "PierSeaSide_3840x2160_2997fps_10bit_420.y4m",
                   "Tango_3840x2160_5994fps_10bit_420.y4m",
                   "TimeLapse_3840x2160_5994fps_10bit_420.y4m"],
"a2_2k"         : ["Aerial3200_1920x1080_5994_10bit_420.y4m",
                   "Boat_1920x1080_5994_10bit_420.y4m",
                   "CrowdRun_1920x1080p50.y4m",
                   "DinnerSceneCropped_1920x1080_2997fps_10bit_420.y4m",
                   "FoodMarket_1920x1080_5994_10bit_420.y4m",
                   "MeridianTalk_sdr_1920x1080p_5994_10bit.y4m",
                   "OldTownCross_1920x1080p50.y4m",
                   "PedestrianArea_1920x1080p25.y4m",
                   "RitualDance_1920x1080_5994_10bit_420.y4m",
                   "Riverbed_1920x1080p25.y4m",
                   "RushFieldCuts_1920x1080_2997.y4m",
                   "Skater227_1920x1080_30fps.y4m",
                   "ToddlerFountain_1920x1080_2997fps_10bit_420.y4m",
                   "TunnelFlag_1920x1080_5994_10bit_420.y4m",
                   "Vertical_bees_1080x1920_2997.y4m",
                   "Vertical_Carnaby_1080x1920_5994.y4m",
                   "WalkingInStreet_1920x1080_30fps.y4m",
                   "WorldCup_1920x1080_30p.y4m",
                   "WorldCup_far_1920x1080_30p.y4m"],
"a3_720p"       : ["ControlledBurn_1280x720p30_420.y4m",
                   "DrivingPOV_1280x720p_5994_10bit_420.y4m",
                   "Johnny_1280x720_60.y4m",
                   "KristenAndSara_1280x720_60.y4m",
                   "RollerCoaster_1280x720p_5994_10bit_420.y4m",
                   "Vidyo3_1280x720p_60fps.y4m",
                   "Vidyo4_1280x720p_60fps.y4m",
                   "WestWindEasy_1280x720p30_420.y4m"],
"a4_360p"       : ["BlueSky_360p25.y4m",
                   "RedKayak_360_2997.y4m",
                   "SnowMountain_640x360_2997.y4m",
                   "SpeedBag_640x360_2997.y4m",
                   "Stockholm_640x360_5994.y4m",
                   "TouchdownPass_640x360_2997.y4m"],
"a5_270p"       : ["FourPeople_480x270_60.y4m",
                   "ParkJoy_480x270_50.y4m",
                   "SparksElevator_480x270p_5994_10bit.y4m",
                   "Vertical_Bayshore_270x480_2997.y4m"],
"b1_syn"        : ["AOV5_1920x1080_60_8bit_420.y4m",
                   "Baolei_2048x1080_60fps.y4m",
                   "CosmosTreeTrunk_sdr_2048x858_25_8bit.y4m",
                   "EuroTruckSimulator2_1920x1080p60.y4m",
                   "GlassHalf_1920x1080p_24p_8bit_420.y4m",
                   "Life_1080p30.y4m",
                   "MINECRAFT_1080p_60_8bit.y4m",
                   "MissionControlClip3_1920x1080_60_420.y4m",
                   "SolLevanteDragons_sdr_1920x1080_24_10bit.y4m",
                   "SolLevanteFace_sdr_1920x1080_24_10bit.y4m",
                   "Wikipedia_1920x1080p30.y4m"],
"hdr1_4k"       : ["MeridianRoad_3840x2160_5994_hdr10.y4m",
                   "NocturneDance_3840x2160_60fps_hdr10.y4m",
                   "NocturneRoom_3840x2160_60fps_hdr10.y4m",
                   "SparksWelding_4096x2160_5994_hdr10.y4m"],
"hdr2_2k"       : ["CosmosCaterpillar_2048x858p24_hdr10.y4m",
                   "CosmosTreeTrunk_2048x858p24_hdr10.y4m",
                   "MeridianShore_1920x1080_5994_hdr10.y4m",
                   "MeridianTalk_1920x1080_5994_hdr10.y4m",
                   "SolLevanteDragons_1920x1080p24_hdr10.y4m",
                   "SolLevanteFace_1920x1080p24_hdr10.y4m",
                   "SparksTruck_2048x1080_5994_hdr10.y4m"],
}
