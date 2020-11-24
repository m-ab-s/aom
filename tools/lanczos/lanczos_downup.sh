#!/bin/bash
#
# Usage:
#   lanczos_downup.sh <input_y4m> <num_frames>
#                     <resample_config_horz>
#                     <resample_config_vert>
#                     <downup_y4m>
#                     [<down_y4m>]
#
#   Notes:
#       <y4m_input> is input y4m video
#       <num_frames> is number of frames to process
#       <horz_resampling_config> is in the format:
#               <horz_p>:<horz_q>:<Lanczos_horz_a>[:horz_x0]
#           similar to what is used by lanczos_resample_y4m utility.
#       <vert_resampling_config> is in the format:
#               <vert_p>:<vert_q>:<Lanczos_vert_a>[:vert_x0]
#           similar to what is used by lanczos_resample_y4m utility.
#       <downup_y4m> is the output y4m video.
#       <down_y4m> provides the intermedite resampled file as an
#           optional parameter. If skipped the intermediate resampled
#           file is deleted.
#

set -e

tmpdir="/tmp"
AOMENC="${tmpdir}/aomenc_$$"
AOMDEC="${tmpdir}/aomdec_$$"
RESAMPLE="${tmpdir}/lanczos_resample_y4m_$$"
cp ./aomenc $AOMENC
cp ./aomdec $AOMDEC
cp ./lanczos_resample_y4m $RESAMPLE

trap 'echo "Exiting..."; rm -f ${AOMENC} ${AOMDEC} ${RESAMPLE}' EXIT

input_y4m=$1
nframes=$2
hdconfig=$3
vdconfig=$4
downup_y4m=$5

#Get width and height
hdr=$(head -1 $input_y4m)
twidth=$(awk -F ' ' '{ print $2 }' <<< "${hdr}")
theight=$(awk -F ' ' '{ print $3 }' <<< "${hdr}")
width=${twidth:1}
height=${theight:1}

#Get intermediate (down) file
if [[ -z $6 ]]; then
  down_y4m=/tmp/down_$$.y4m
else
  down_y4m=$6
fi

#Obtain the horizontal and vertical upsampling configs
hdconfig_arr=(${hdconfig//:/ })
huconfig="${hdconfig_arr[1]}:${hdconfig_arr[0]}:${hdconfig_arr[2]}"
if [[ -n ${hdconfig_arr[3]} ]]; then
  huconfig="${huconfig}:i${hdconfig_arr[3]}"
fi
vdconfig_arr=(${vdconfig//:/ })
vuconfig="${vdconfig_arr[1]}:${vdconfig_arr[0]}:${vdconfig_arr[2]}"
if [[ -n ${vdconfig_arr[3]} ]]; then
  vuconfig="${vuconfig}:i${vdconfig_arr[3]}"
fi

$RESAMPLE $input_y4m $nframes $hdconfig $vdconfig $down_y4m &&
$RESAMPLE $down_y4m $nframes $huconfig $vuconfig $downup_y4m ${width}x${height}

#tiny_ssim_highbd $input_y4m $downup_y4m

if [[ -z $6 ]]; then
  rm $down_y4m
fi
