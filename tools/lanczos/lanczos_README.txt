/*
 * Copyright (c) 2020, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

The Lanczos resampling library and utilities are developed by Google
initially and contributed to the Alliance for open Media for experiments
related to super-resolution and Adaptive Streaming use cases.

For questions and technical support, please contact debargha@google.com

LIBRARY
#######

The main library source and header files are:
lanczos_resample.c
lanczos_resample.h


UTILITIES
#########

In addition, two utilities are provided to resample y4m and yuv videos
respectively:
lanczos_resample_y4m (built using source lanczos_resample_y4m.c)
lanczos_resample_y4m (built using source lanczos_resample_yuv.c)

lanczos_resample_y4m
--------------------
The usage of lanczos_resample_y4m is:
  lanczos_resample_y4m
      <y4m_input>
      <num_frames>
      <horz_p>:<horz_q>:<Lanczos_horz_a>[:<horz_x0>:<horz_ext>]
      <vert_p>:<vert_q>:<Lanczos_vert_a>[:<vert_x0>:<vert_ext>]
      <y4m_output>
      [<outwidth>x<outheight>]
  Notes:
      <num_frames> is number of frames to be processed
      <horz_p>/<horz_q> gives the horz resampling ratio.
      <vert_p>/<vert_q> gives the vert resampling ratio.
      <Lanczos_horz_a>, <Lanczos_vert_a> are Lanczos parameters.
      <horz_x0>, <vert_x0> are optional initial offsets
                                        [default centered].
          If used, they can be a number in (-1, 1),
                   or a number in (-1, 1) prefixed by 'i' meaning
                       using the inverse of the number provided,
                   or 'c' meaning centered
      <horz_ext>, <vert_ext> are optional extension types:
                   'r' (Repeat) or 's' (Symmetric) [default: 'r']
      <outwidth>x<outheight> is output video dimensions
                             only needed in case of upsampling
      Resampling config string of 1:1:1:0 horizontally or vertically
          is regarded as a no-op in that direction.

Example usages:

1a. Downsample 10 frames from Boat_1920x1080_60fps_10bit_420.y4m by
    ratio 3/2 horizontally and 4/3 vertically with Lanczos parameter
    6 for both and centered sampling.

    lanczos_resample_y4m Boat_1920x1080_60fps_10bit_420.y4m 20 \
                         2:3:6 3:4:6 /tmp/down.y4m

1b. Reverse the process in 1a. That is, upsample 10 frames from
    /tmp/boat.y4m by ratio 3/2 horizontally and 4/3 vertically with
    Lanczos parameter 6 for both and centered sampling. The output
    dimension desired is specified to be 1920x1080.

    lanczos_resample_y4m /tmp/down.y4m 10 3:2:6 4:3:6 /tmp/downup.y4m 1920x1080

2a. Similar to 1a, except using left-aligned sampling

    lanczos_resample_y4m Boat_1920x1080_60fps_10bit_420.y4m 20 \
                         2:3:6:0 3:4:6:0 /tmp/down.y4m

2b. Reversing the process in 2a.

    lanczos_resample_y4m /tmp/down.y4m 10 3:2:6:0 4:3:6:0 /tmp/downup.y4m \
                         1920x1080

3a. Similar to 1a, except using a specified initial offset of 0.125 horizontally
    and center aligned sampling vertically

    lanczos_resample_y4m Boat_1920x1080_60fps_10bit_420.y4m 20 \
                         2:3:6:0.125 3:4:6 /tmp/down.y4m

3b. Reversing the process in 3a.

    lanczos_resample_y4m /tmp/down.y4m 10 3:2:6:i0.125 4:3:6 /tmp/downup.y4m \
                         1920x1080

4a. Similar to 3a, except using symmetric border extension for vertical
    resampling instead of default repeat.

    lanczos_resample_y4m Boat_1920x1080_60fps_10bit_420.y4m 20 \
                         2:3:6:0.125 3:4:6:c:s /tmp/down.y4m

4b. Reversing the process in 4a.

    lanczos_resample_y4m /tmp/down.y4m 10 3:2:6:i0.125 4:3:6:c:s \
                         /tmp/downup.y4m 1920x1080

lanczos_resample_yuv
--------------------
The usage of lanczos_resample_yuv is similar but with two extra
arguments to specify the input format:
  lanczos_resample_yuv
      <yuv_input>
      <width>x<height>
      <pix_format>
      <num_frames>
      <horz_p>:<horz_q>:<Lanczos_horz_a>[:<horz_x0>:<horz_ext>]
      <vert_p>:<vert_q>:<Lanczos_vert_a>[:<vert_x0>:<vert_ext>]
      <yuv_output>
      [<outwidth>x<outheight>]
  Notes:
      <width>x<height> is input video dimensions.
      <pix_format> is one of { yuv420p, yuv420p10, yuv420p12,
                               yuv422p, yuv422p10, yuv422p12,
                               yuv444p, yuv444p10, yuv444p12 }
      <num_frames> is number of frames to be processed
      <horz_p>/<horz_q> gives the horz resampling ratio.
      <vert_p>/<vert_q> gives the vert resampling ratio.
      <Lanczos_horz_a>, <Lanczos_vert_a> are Lanczos parameters.
      <horz_x0>, <vert_x0> are optional initial offsets
                                        [default centered].
          If used, they can be a number in (-1, 1),
                   or a number in (-1, 1) prefixed by 'i' meaning
                       using the inverse of the number provided,
                   or 'c' meaning centered
      <horz_ext>, <vert_ext> are optional extension types:
                   'r' (Repeat) or 's' (Symmetric) [default: 'r']
      <outwidth>x<outheight> is output video dimensions
                             only needed in case of upsampling
      Resampling config string of 1:1:1:0 horizontally or vertically
          is regarded as a no-op in that direction.


SCRIPTS
#######

In addition to these utilities, two convenience scripts are provided:
lanczos_downup.sh
lanczos_downcompup.sh
They can be invoked from the build directory directly after a build that
produces lanczos_resample_y4m, aomenc and aomdec applications.

lanczos_downup.sh
-----------------
The script lanczos_downup.sh resamples a video with specified parameters
and then reverses the process using two lanczos_resample_y4m commands.
The usage for the script is:

  lanczos_downup.sh
      <y4m_input>
      <num_frames>
      <horz_resampling_config>
      <vert_resampling_config>
      <downup_y4m>
      [<down_y4m>]

  Notes:
      <y4m_input> is input y4m video
      <num_frames> is number of frames to process
      <horz_resampling_config> is in the format:
              <horz_p>:<horz_q>:<Lanczos_horz_a>[:<horz_x0>:<horz_ext>]
          similar to what is used by lanczos_resample_y4m utility.
      <vert_resampling_config> is in the format:
              <vert_p>:<vert_q>:<Lanczos_vert_a>[:<vert_x0>:<vert_ext>]
          similar to what is used by lanczos_resample_y4m utility.
      <downup_y4m> is the output y4m video.
      <down_y4m> provides the intermedite resampled file as an
          optional parameter. If skipped the intermediate resampled
          file is deleted.

Example usages:
5. Similar to use case 1a and 1b above.

  From build directory run:
  /path/to/script/lanczos_downup.sh Boat_1920x1080_60fps_10bit_420.y4m \
      20 2:3:6 3:4:6 /tmp/downup.y4m
          [Here the intermediate resampled files is not stored]

  From build directory run:
  /path/to/script/lanczos_downup.sh Boat_1920x1080_60fps_10bit_420.y4m \
      20 2:3:6 3:4:6 /tmp/downup.y4m \
      /tmp/down.y4m
          [Here the intermediate resampled files is stored in /tmp/down.y4m]

lanczos_downcompup.sh
---------------------
The script lanczos_downcompup.sh resamples a video with specified parameters
using lanczos_resample_y4m, then comprsses and decompresses using aomenc and
aomdec respectively, and finally reverse resamples the decompressed video to
the source resolution using another lanczos_resample_y4m command.
The usage for the script is:

  lanczos_downcompup.sh <input_y4m> <num_frames>
                        <resample_config_horz>
                        <resample_config_vert>
                        <cq_level>[:<cpu_used>]
                        <downcompup_y4m>
                        [[<down_y4m>]:[<downcomp_bit>]:[<downcomp_y4m]]

  Notes:
      <y4m_input> is input y4m video
      <num_frames> is number of frames to process
      <horz_resampling_config> is in the format:
              <horz_p>:<horz_q>:<Lanczos_horz_a>[:<horz_x0>:<horz_ext>]
          similar to what is used by lanczos_resample_y4m utility.
      <vert_resampling_config> is in the format:
              <vert_p>:<vert_q>:<Lanczos_vert_a>[:<vert_x0>:<vert_ext>]
          similar to what is used by lanczos_resample_y4m utility.
      <cq_level>[:<cpu_used>] provides the cq_level parameter of
          compression along with an optional cpu_used parameter.
      <downcompup_y4m> is the output y4m video.
      The last param [[<down_y4m>]:[<downcomp_bit>]:[<downcomp_y4m]]
          provides names of intermediate files where:
	      down_y4m is the resampled source
	      downcomp_bit is the compressed resampled bitstream
	      downcomp_y4m is the reconstructed bitstream.
          This parameter string is entirely optional.
          Besides if provided, each of down_y4m, downcomp_bit and
          downcomp_y4m are optional by themselves where each can be
          either provided or empty. If empty the corresponding
	  intermediate file is deleted.

Example usages:
6. Similar to use case 1a and 1b above with a compression step in between.

  From build directory run:
  /path/to/script/lanczos_downcompup.sh Boat_1920x1080_60fps_10bit_420.y4m \
      20 2:3:6 3:4:6 40:5 /tmp/downup.y4m
          [Here no intermediate files are stored]

  From build directory run:
  /path/to/script/lanczos_downcompup.sh Boat_1920x1080_60fps_10bit_420.y4m \
      20 2:3:6 3:4:6 40:5 /tmp/downup.y4m \
      /tmp/down.y4m::/tmp/downrec.y4m
          [Here the resampled source and its compressed reconstruction are
           stoted in /tmp/down.y4m and /tmp/downrec.y4m respectively].

  From build directory run:
  /path/to/script/lanczos_downcompup.sh Boat_1920x1080_60fps_10bit_420.y4m \
      20 2:3:6 3:4:6 40:5 /tmp/downup.y4m \
      /tmp/down.y4m:/tmp/down.bit:/tmp/downrec.y4m
          [Here the resampled source, its compressed bitstream, and the
	   conpressed reconstruction are stoted in /tmp/down.y4m,
	   /tmp/downcomp.bit and /tmp/downrec.y4m respectively].
