import math
import numpy
import csv
import datetime

from mantid.kernel import ConfigService
from mantid.simpleapi import RenameWorkspace, SaveRKH, SaveNXcanSAS, GroupWorkspaces, mtd, ConjoinWorkspaces, SaveNexusProcessed
from mantid import config
from sans.user_file.toml_parsers.toml_reader import TomlReader
import sans.command_interface.ISISCommandInterface as ici

# Setup by rundetection
user_file = "/extras/sans2d/MaskFile.toml"
sample_scatter = 110754
sample_transmission = None
sample_direct = None
can_scatter = None
can_transmission = None
can_direct = None
sample_thickness = 1.0
sample_geometry = "Disc"
sample_height = 8.0
sample_width = 8.0
slice_wavs = [1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 8.75, 10.75, 12.5]
phi_limits_list = [(-30, 30), (60, 120)]

# Other configuration options
output_path = f"/output/run-{sample_scatter}/"
config['defaultsave.directory'] = output_path
import os
print("Before masks check")
if os.path.exists("/archive/NDXSANS2D/user/Masks/"):
    print("/archive/NDXSANS2D/user/Masks/ exists")
    print(os.listdir("/archive/NDXSANS2D/user/Masks/"))
else:
    print("/archive/NDXSANS2D/user/masks/ exists")
    print(os.listdir("/archive/NDXSANS2D/user/masks/"))
ConfigService.setDataSearchDirs("/archive/NDXSANS2D/User/Masks/")

output = []

# Setup ISIS Command Interface for 1D
def cleanup_and_setup_ici():
    ici.Clean()
    ici.UseCompatibilityMode()
    ici.SANS2D()
    ici.Set1D()
    ici.MaskFile(user_file)
    ici.AssignSample(str(sample_scatter))
    if sample_transmission is not None and sample_direct is not None:
        ici.TransmissionSample(str(sample_transmission), str(sample_direct))
    if can_scatter is not None:
        ici.AssignCan(str(can_scatter))
    if can_scatter is not None and can_transmission is not None and can_direct is not None:
        ici.TransmissionCan(str(can_transmission), str(can_direct))


def get_nxcansas_kwargs(workspace_name: str, sector: bool = False, ws_suffix: str | None = None):
    if ws_suffix is None:
        # Get the last 2 numbers from workspace and add them to suffix
        if not sector:
            workspace_name_split = workspace_name.split("_")
        else:
            workspace_name_split = workspace_name.split("Phi")[0].split("_")
        ws_suffix = f"{workspace_name_split[-2]}_{workspace_name_split[-1]}"
    kwargs = {}
    kwargs["InputWorkspace"] = workspace_name
    kwargs["Filename"] = f"{output_path}/{workspace_name}_auto.h5"
    kwargs["Geometry"] = sample_geometry
    kwargs["SampleHeight"] = sample_height
    kwargs["SampleWidth"] = sample_width
    kwargs["SampleThickness"] = sample_thickness
    if "merged" in workspace_name.lower():
        kwargs["DetectorNames"] = "front-detector,rear-detector"
    elif "front" in workspace_name.lower():
        kwargs["DetectorNames"] = "front-detector"
    elif "rear" in workspace_name.lower():
        kwargs["DetectorNames"] = "rear-detector"
    if sample_transmission is not None and sample_direct is not None:
        kwargs["Transmission"] = f"{sample_scatter}_trans_Sample_{ws_suffix}"
        kwargs["SampleTransmissionRunNumber"] = str(sample_transmission)
        kwargs["SampleDirectRunNumber"] = str(sample_direct)
    if can_scatter is not None:
        kwargs["CanScatterRunNumber"] = str(can_scatter)
    if can_scatter is not None and can_transmission is not None and can_direct is not None:
        kwargs["TransmissionCan"] = f"{sample_scatter}_trans_Can_{ws_suffix}"
        kwargs["CanDirectRunNumber"] = str(can_direct)
    return kwargs


# Perform 1D reduction and output
cleanup_and_setup_ici()
output_workspace_1d_merged = ici.WavRangeReduction(None, None, ici.DefaultTrans, combineDet="merged")
output_workspace_1d_front = output_workspace_1d_merged.replace("merged", "front")
output_workspace_1d_rear = output_workspace_1d_merged.replace("merged", "rear")
for output_workspace in [output_workspace_1d_merged, output_workspace_1d_front, output_workspace_1d_rear]:
    SaveNXcanSAS(**get_nxcansas_kwargs(output_workspace))
    output.append(f"{output_workspace}_auto.h5")
    SaveRKH(output_workspace, f"{output_path}/{output_workspace}_auto.dat")
    output.append(f"{output_workspace}_auto.dat")

# Setup ISIS Command Interface for 2D and reduce
ici.Set2D()
output_workspace_2d_merged = ici.WavRangeReduction(None, None, ici.DefaultTrans, combineDet="merged")
output_workspace_2d_front = output_workspace_2d_merged.replace("merged", "front")
output_workspace_2d_rear = output_workspace_2d_merged.replace("merged", "rear")
for output_workspace in [output_workspace_2d_merged, output_workspace_2d_front, output_workspace_2d_rear]:
    SaveNXcanSAS(**get_nxcansas_kwargs(output_workspace))
    output.append(f"{output_workspace}_auto.h5")


def save_sector_reduction(output_workspace):
    # Assume merged
    def save_out_ws(output_workspace_inside):
        SaveNXcanSAS(**get_nxcansas_kwargs(output_workspace_inside, sector=True))
        output.append(output_workspace_inside + "_auto")

    output_workspace_front = output_workspace.replace("merged", "front")
    output_workspace_rear = output_workspace.replace("merged", "rear")
    save_out_ws(output_workspace)
    save_out_ws(output_workspace_rear)
    save_out_ws(output_workspace_front)


# Now perform the sector reduction
cleanup_and_setup_ici()
for phi_limits in phi_limits_list:
    ici.SetPhiLimit(phi_limits[0], phi_limits[1], use_mirror=True)
    output_workspaces = ici.WavRangeReduction(None, None, ici.DefaultTrans, combineDet="merged")
    save_sector_reduction(output_workspaces)


# Now perform the overlap reduction
cleanup_and_setup_ici()
ici.Set1D()
# Create a multiple scattering workspaces to check for well, multiple scattering.
ms_workspaces = []
for i in range(len(slice_wavs) - 1):
  output_workspace = ici.WavRangeReduction(slice_wavs[i], slice_wavs[i + 1], False, combineDet='merged')
  ms_workspaces.append(output_workspace)

ms_ws = f"{sample_scatter}_merged_multiple_scattering"
ConjoinWorkspaces(InputWorkspace1=ms_workspaces[0], InputWorkspace2=ms_workspaces[1], CheckOverlapping=False)
RenameWorkspace(InputWorkspace=ms_workspaces[0], OutputWorkspace=ms_ws)
for ws_index in range(len(ms_workspaces) - 1):
    if ws_index not in [0, 1]:
        ConjoinWorkspaces(InputWorkspace1=ms_ws, InputWorkspace2=ms_workspaces[ws_index], CheckOverlapping=False)
SaveNexusProcessed(InputWorkspace=ms_ws, Filename=f"{ms_ws}.nxs")


# Create batch csv file for loading into mantid for manual equivalent reduction
first_line = f"# MANTID_BATCH_FILE created on {datetime.datetime.now(tz=datetime.timezone.utc)} by FIA (Automated reduction)\n"
with open(f"{output_path}/batch.csv", "w") as csvfile:
    csvfile.write(first_line)
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    row = []
    for item in ["sample_sans",sample_scatter,"sample_trans",sample_transmission,"sample_direct_beam",sample_direct,"can_sans",can_scatter,"can_trans",can_transmission,"can_direct_beam",can_direct,"output_as",f"run-{sample_scatter}"]:
        if item is not None:
            row.append(item)
        else:
            # Remove the already added string from before
            row.pop()
    writer.writerow(row)
