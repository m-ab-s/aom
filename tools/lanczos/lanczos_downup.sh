#!/bin/bash
#
# Usage:
#   lanczos_downup.sh <input_y4m> <num_frames>
#                     <resample_config_horz> <resample_config_vert>
#                     <downup_y4m> [<down_y4m>]
#             down_y4m is optional.
#

input_y4m=$1
nframes=$2
hdconfig=$3
vdconfig=$4

hdr=$(head -1 $input_y4m)
twidth=$(awk -F ' ' '{ print $2 }' <<< "${hdr}")
theight=$(awk -F ' ' '{ print $3 }' <<< "${hdr}")
width=${twidth:1}
height=${theight:1}

downup_y4m=$5

if [[ -z $6 ]]; then
  down_y4m=/tmp/down_$$.y4m
else
  down_y4m=$6
fi

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

./lanczos_resample_y4m $input_y4m $nframes $hdconfig $vdconfig $down_y4m &&
./lanczos_resample_y4m $down_y4m $nframes $huconfig $vuconfig \
    $downup_y4m ${width}x${height}

#tiny_ssim_highbd $input_y4m $downup_y4m

if [[ -z $6 ]]; then
  rm $down_y4m
fi
