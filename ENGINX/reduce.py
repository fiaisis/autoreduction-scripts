from pathlib import Path

from mantid.simpleapi import *
from Engineering.EnginX import EnginX
from Engineering.EnggUtils import GROUP


# Values changed by rundetection
vanadium_run = "ENGINX236516"
focus_runs = ["ENGINX299080"]
ceria_run = "ENGINX193749"
group = GROUP["BOTH"]

# Set values that don't change
output_dir = f"/output/run-{focus_runs[0]}"
calib_file = "/opt/conda/envs/mantid/scripts/Engineering/calib/ENGINX_full_instrument_calibration_193749.nxs"

output = []  # This is probably the nexus files in the focus dir or something
enginx = EnginX(
    vanadium_run=vanadium_run,
    focus_runs=focus_runs,
    save_dir=output_dir,
    full_inst_calib_path=calib_file,
    ceria_run=ceria_run,
    group=group,
)
enginx.main(plot_cal=False, plot_foc=False)

for path in Path(output_dir).rglob("*"):
    if path.is_file():
        output.append(str(path))
