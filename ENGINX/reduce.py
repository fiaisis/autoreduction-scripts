from pathlib import Path

from mantid.simpleapi import *
from Engineering.EnginX import EnginX
from Engineering.EnggUtils import GROUP


# Values changed by rundetection
vanadium_path = "some path"
focus_path = "some path"
ceria_path = "some path"
group = GROUP["BOTH"]

print(f"Starting run with focus {focus_path}, vanadium {vanadium_path}, and ceria {ceria_path}.")

# Set values that don't change
folder = focus_path.split("/")[-1].split(".")[0]
output_dir = "/output"
calib_file = "/opt/conda/envs/mantid/scripts/Engineering/calib/ENGINX_full_instrument_calibration_193749.nxs"
output = []
enginx = EnginX(
    vanadium_run=vanadium_path,
    focus_runs=[focus_path],
    save_dir=output_dir,
    full_inst_calib_path=calib_file,
    ceria_run=ceria_path,
    group=group,
)
enginx.main(plot_cal=False, plot_foc=False)

for path in Path(output_dir).rglob("*"):
    if path.is_file():
        output.append(str(path.name))
