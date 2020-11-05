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

#TEST_SET = ["a1_4k", "a2_2k", "a3_720p", "a4_360p", "a5_270p", "b1_syn", "hdr"]
CTC_TEST_SET = ["a5_270p"]
AS_TEST_SET = ["a1_4k"]
Y4M_CLIPs = {
"a1_4k"         : ["Crosswalk_3840x2160_5994fps_10bit_420.y4m"],
'''
"a1_4k"         : ["BoxingPractice_3840x2160_5994fps_10bit_420.y4m", "Crosswalk_3840x2160_5994fps_10bit_420.y4m",
                   "FoodMarket2_3840x2160_5994fps_10bit_420.y4m", "Neon1224_3840x2160_2997fps.y4m",
                   "NocturneDance_3840x2160p_10bit_60fps.y4m", "PierSeaSide_3840x2160_2997fps_10bit_420.y4m",
                   "Tango_3840x2160_5994fps_10bit_420.y4m", "TimeLapse_3840x2160_5994fps_10bit_420.y4m"],
'''
"a2_2k"         : ["Aerial3200_1920x1080_5994_10bit_420.y4m", "Boat_1920x1080_60fps_10bit_420.y4m",
                   "crowd_run_1080p50.y4m", "FoodMarket_1920x1080_60fps_10bit_420.y4m",
                   "Meridian_talk_sdr_1080p_10bit.y4m", "old_town_cross_1080p50.y4m",
                   "pedestrian_area_1080p25.y4m", "raw_1080x1920@30.walking.in.street.y4m",
                   "RitualDance_1920x1080_60fps_10bit_420.y4m", "riverbed_1080p25.y4m",
                   "rush_field_cuts_1080p.y4m", "Skater227_1920x1080_30fps.y4m",
                   "TunnelFlag_1920x1080_60fps_10bit_420.y4m", "Vertical_bees_2997_1080x1920.y4m",
                   "Vertical_Carnaby_5994_1080x1920.y4m", "WorldCup_1920x1080_30p.y4m",
                   "WorldCup_far_1920x1080_30p.y4m"],
"a3_720p"       : ["controlled_burn_720p_420.y4m", "Johnny_1280x720_60.y4m",
                   "KristenAndSara_1280x720_60.y4m", "Netflix_DrivingPOV_720p_60fps_10bit_420.y4m",
                   "Netflix_RollerCoaster_720p_60fps_10bit_420.y4m", "vidyo3_720p_60fps.y4m",
                   "vidyo4_720p_60fps.y4m", "west_wind_easy_720p_420.y4m"],
"a4_360p"       : ["blue_sky_360p25.y4m", "red_kayak_360p.y4m",
                   "snow_mnt_640x360p.y4m", "speed_bag_640x360p.y4m",
                   "stockholm_640x360p60.y4m", "touchdown_pass_640x360p30.y4m"],
"a5_270p"       : ["FourPeople_480x270_60.y4m", "park_joy_480x270_50.y4m",
                   "sparks_elevator_480x270p_60.y4m", "Vertical_Bayshore_2997_270x480.y4m"],
"b1_syn"        : ["AOV5_1920x1080_60_8bit_420.y4m", "baolei_2048x1080_60fps.y4m",
                   "cosmos_log_2048x858_25_8bit_sdr.y4m", "EuroTruckSimulator2.y4m",
                   "GlassHalf_1920x1080p_24p_8b_420.y4m", "life_1080p30.y4m",
                   "MINECRAFT_1080p_60_8bit.y4m", "MissionControlClip3_1920x1080_60_420.y4m",
                   "sol_levante_dragons_sdr_1920x1080_24_10bit_full.y4m", "sol_levante_face_sdr_1920x1080_24_10bit.y4m",
                   "wikipedia_1920x1080p30.y4m"],
"hdr"           : ["cosmos_caterpillar_2048x858p24_hdr10.y4m", "cosmos_tree_trunk_2048x858p24_hdr10.y4m",
                   "meridian_shore_1920x1080p60_hdr10.y4m", "meridian_talk_1920x1080p60_hdr10.y4m",
                   "merid_road_3840x2160p60_hdr10.y4m", "nocturne_dance_3840x2160_hdr10.y4m",
                   "nocturne_room_3840x2160_hdr10.y4m", "sol_levante_dragons_1920x1080p24_hdr10.y4m",
                   "sol_levante_face_1920x1080p24_hdr10.y4m", "sparks_truck_2048x1080p60_hdr10.y4m",
                   "sparks_welding_4096x2160p60_hdr10.y4m"],
}
