# Vivado HLS / Vitis HLS build script for loeffler_dct_8pt IP
# Usage: vitis_hls -f build_hls.tcl

open_project l2l_dct_hls
set_top loeffler_dct_8pt
add_files loeffler_dct_8pt.cpp
add_files -tb tb_loeffler.cpp
open_solution "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
export_design -format ip_catalog
exit
