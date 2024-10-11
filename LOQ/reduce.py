from mantid.kernel import ConfigService
import math
import numpy
from mantid.simpleapi import RenameWorkspace, SaveRKH, SaveNXcanSAS, GroupWorkspaces, mtd
from mantid import config
from sans.user_file.toml_parsers.toml_reader import TomlReader
import sans.command_interface.ISISCommandInterface as ici

# Setup by rundetection
user_file = "/extras/loq/MaskFile.toml"
sample_scatter = 110754 # Will need the 00 added for new cycles
sample_transmission = None
sample_direct = None
can_scatter = None
can_transmission = None
can_direct = None
sample_thickness = 1.0
sample_geometry = "Disc"
sample_height = 8.0
sample_width = 8.0

# Other configuration options
output_path = f"/output/run-{sample_scatter}/"
config['defaultsave.directory'] = output_path
ConfigService.setDataSearchDirs("/archive/NDXLOQ/user/masks/")
default_slice_wavs = [2.7, 3.7, 4.7, 5.7, 6.7, 8.7, 10.5]

output = []

# Setup ISIS Command Interface for 1D
def cleanup_and_setup_ici():
  ici.Clean()
  ici.UseCompatibilityMode()
  ici.LOQ()
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
    kwargs["DetectorNames"] = "main-detector-bank,HAB"
  elif "hab" in workspace_name.lower():
    kwargs["DetectorNames"] = "HAB"
  elif "lab" in workspace_name.lower():
    kwargs["DetectorNames"] = "main-detector-bank"
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
output_workspace_1d_hab = output_workspace_1d_merged.replace("merged", "HAB")
output_workspace_1d_lab = output_workspace_1d_merged.replace("merged", "main")
for output_workspace in [output_workspace_1d_merged, output_workspace_1d_hab, output_workspace_1d_lab]:
  SaveNXcanSAS(**get_nxcansas_kwargs(output_workspace))
  output.append(f"{output_workspace}_auto.h5")
  SaveRKH(output_workspace, f"{output_path}/{output_workspace}_auto.dat")
  output.append(f"{output_workspace}_auto.dat")

# Setup ISIS Command Interface for 2D and reduce
ici.Set2D()
output_workspace_2d_merged = ici.WavRangeReduction(None, None, ici.DefaultTrans, combineDet="merged")
output_workspace_2d_hab = output_workspace_2d_merged.replace("merged", "HAB")
output_workspace_2d_lab = output_workspace_2d_merged.replace("merged", "main")
for output_workspace in [output_workspace_2d_merged, output_workspace_2d_hab, output_workspace_2d_lab]:
  SaveNXcanSAS(**get_nxcansas_kwargs(output_workspace))
  output.append(f"{output_workspace}_auto.h5")

def save_sector_reduction(output_workspace, sector):
  # Assume merged
  def save_out_ws(output_workspace_inside, sector):
      RenameWorkspace(output_workspace_inside, output_workspace_inside + sector)
      SaveNXcanSAS(**get_nxcansas_kwargs(output_workspace_inside + sector, sector=True))
      output.append(output_workspace_inside + sector + "_auto")
  output_workspace_hab = output_workspace.replace("merged", "HAB")
  output_workspace_lab = output_workspace.replace("merged", "main")
  save_out_ws(output_workspace, sector)
  save_out_ws(output_workspace_lab, sector)
  save_out_ws(output_workspace_hab, sector)


# Now perform the sector reduction
cleanup_and_setup_ici()
ici.SetPhiLimit(-30, 30, use_mirror=True)
output_workspaces = ici.WavRangeReduction(None, None, ici.DefaultTrans, combineDet="merged")
save_sector_reduction(output_workspaces, "_horizontal_sector")

ici.SetPhiLimit(60, 120, use_mirror=True)
output_workspaces=ici.WavRangeReduction(None, None, ici.DefaultTrans, combineDet="merged")
save_sector_reduction(output_workspaces, "_vertical_sector")

# Now perform the overlap reduction
cleanup_and_setup_ici()
ici.Set1D()
for i in range(len(default_slice_wavs) - 1):
  output_workspace = ici.WavRangeReduction(default_slice_wavs[i], default_slice_wavs[i + 1], False, combineDet='merged')
  SaveNXcanSAS(**get_nxcansas_kwargs(output_workspace, ws_suffix=f"{default_slice_wavs[0]}_{default_slice_wavs[-1]}"))
  output.append(f"{output_workspace}_auto.h5")
