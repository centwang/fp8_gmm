#!/bin/bash

sudo -E env "PATH=$PATH" /opt/nvidia/nsight-compute/2024.1.0/ncu --export $1 \
--section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis \
--section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy \
--section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart \
--section WarpStateStats --force-overwrite --target-processes all --replay-mode kernel \
--launch-skip-before-match 0 \
--nvtx --profile-from-start 1 --cache-control all --clock-control none --apply-rules no \
--check-exit-code yes  --import-source yes --kernel-name regex:$2 ${@:3}
