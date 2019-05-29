#!/bin/bash
#
# wrapper script to run HCC_PROFILE=2 and summarize report in with RPT
#
HCC_PROFILE=${HCC_PROFILE:="hcc_profile.txt"}
RPT_PROFILE=${RPT_PROFILE:="rpt.txt"}

env HCC_PROFILE=2 $@ 2>${HCC_PROFILE}
/opt/rocm/hcc/bin/rpt ${HCC_PROFILE} > ${RPT_PROFILE}
grep busy ${RPT_PROFILE} | head -1 | awk '{ print $5, " ", $6 }'
