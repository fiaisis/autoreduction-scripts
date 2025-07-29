# import mantid algorithms, numpy and matplotlib
from pathlib import Path

from mantid.simpleapi import *
from Engineering.EnginX import EnginX
from Engineering.EnggUtils import GROUP

FULL_CALIB = "/opt/conda/envs/mantid/scripts/Engineering/calib/ENGINX_full_instrument_calibration_193749.nxs"

vanadium_run = "ENGINX236516"
focus_runs = ["ENGINX299080"]
ceria_run = "ENGINX193749"
group = GROUP["BOTH"]

run_dir = [focus_runs[0][6:]]
CWDIR = f"/output/run-{run_dir}"

output = []  # This is probably the nexus files in the focus dir or something
enginx = EnginX(
    vanadium_run=vanadium_run,
    focus_runs=focus_runs,
    save_dir=CWDIR,
    full_inst_calib_path=FULL_CALIB,
    ceria_run=ceria_run,
    group=GROUP.BOTH,
)
enginx.main(plot_cal=False, plot_foc=False)

for path in Path(CWDIR).rglob("*"):
    if path.is_file():
        output.append(str(path))
