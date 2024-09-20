import math
import numpy
from mantid.simpleapi import RenameWorkspace, SaveRKH, SaveNXcanSAS, GroupWorkspaces, SaveNexusProcessed
from mantid import config
from sans.user_file.toml_parsers.toml_reader import TomlReader
import sans.command_interface.ISISCommandInterface as ici

# Setup by rundetection
user_file = "/extras/loq/MaskFile.toml"
sample_scatter = "74044"  # Will need the 00 added for new cycles
sample_transmission = "74024"
sample_direct = "74014"
can_scatter = "74019"
can_transmission = "74020"
can_direct = "74014"

# Save and reduction options
output_path = f"/output/run-{sample_scatter}/"
config['defaultsave.directory'] = output_path


def grab_wavelength_limits_from_user_file():
    # Will result in an extra parsing of the file but needed here.
    read_toml = TomlReader.get_user_file_dict(user_file)
    wavelength_details = read_toml['binning']['wavelength']
    step = wavelength_details['step']
    start = wavelength_details['start']
    stop = wavelength_details['stop']
    if wavelength_details['type'] == "Log":
        log_step = 1 + step
        wavs = [start]
        while True:
            next_value = wavs[-1] * log_step
            if next_value > stop:
                break
            else:
                wavs.append(next_value)
        wavs[-1] = stop
        return wavs
    elif wavelength_details['type'] == "Lin":
        return list(numpy.arange(start=start, stop=stop, step=step))
    else:
        raise ValueError(f"User file does not contain Log or Lin ranges for wavelengths uses type "
                         f"{wavelength_details['type']}")


wavs = grab_wavelength_limits_from_user_file()

output = []

# Setup ISIS Command Interface for 1D
def cleanup_and_set_ici():
    ici.Clean()
    ici.UseCompatibilityMode()
    ici.LOQ()
    ici.Set1D()
    ici.MaskFile(user_file)
    ici.AssignSample(sample_scatter)
    if sample_transmission is not None and sample_direct is not None:
        ici.TransmissionSample(sample_transmission, sample_direct)
    if can_scatter is not None:
        ici.AssignCan(can_scatter)
    if can_scatter is not None and can_transmission is not None and can_direct is not None:
        ici.TransmissionCan(can_transmission, can_direct)

# Perform 1D reduction and output
cleanup_and_set_ici()
output_workspace_1d_merged = ici.WavRangeReduction(None, None, ici.DefaultTrans, combineDet="merged")
output_workspace_1d_hab = output_workspace_1d_merged.replace("merged", "HAB")
output_workspace_1d_lab = output_workspace_1d_merged.replace("merged", "main")
for output_workspace in [output_workspace_1d_merged, output_workspace_1d_hab, output_workspace_1d_lab]:
    SaveNXcanSAS(output_workspace, f"{output_path}/{output_workspace}.h5")
    output.extend(f"{output_workspace}.h5")
    SaveRKH(output_workspace, f"{output_path}/{output_workspace}.dat")
    output.extend(f"{output_workspace}.dat")

# Setup ISIS Command Interface for 2D and reduce
ici.Set2D()
output_workspace_2d_merged = ici.WavRangeReduction(None, None, ici.DefaultTrans, combineDet="merged")
output_workspace_2d_hab = output_workspace_2d_merged.replace("merged", "HAB")
output_workspace_2d_lab = output_workspace_2d_merged.replace("merged", "main")
for output_workspace in [output_workspace_2d_merged, output_workspace_2d_hab, output_workspace_2d_lab]:
    SaveNXcanSAS(output_workspace, f"{output_path}/{output_workspace}.h5")
    output.extend(f"{output_workspace}.h5")

# Now perform the overlap reduction
ici.Set1D()
output_workspaces_overlap_reductions_name_string = ""
for i in range(len(wavs) - 1):
    output_workspaces_overlap_reductions_name_string += f", {ici.WavRangeReduction(wavs[i], wavs[i + 1], False, combineDet='merged')}"
overlap_reduction_name = "overlap_reduction_group_workspaces"
GroupWorkspaces(InputWorkspaces=output_workspaces_overlap_reductions_name_string, OutputWorkspace=overlap_reduction_name)
SaveNexusProcessed(overlap_reduction_name, f"{output_path}/{overlap_reduction_name}.nxs")
output.extend(overlap_reduction_name)


def save_sector_reduction(output_workspace, sector):
    # Assume merged
    def save_out_ws(output_workspace_inside, sector):
        RenameWorkspace(output_workspace_inside, output_workspace_inside + sector)
        SaveNXcanSAS(output_workspace_inside + sector, f"{output_path}/{output_workspace_inside}{sector}.h5")
        output.append(output_workspace_inside + sector)
    output_workspace_hab = output_workspace.replace("merged", "HAB")
    output_workspace_lab = output_workspace.replace("merged", "main")
    save_out_ws(output_workspace, sector)
    save_out_ws(output_workspace_lab, sector)
    save_out_ws(output_workspace_hab, sector)


# Now perform the sector reduction
cleanup_and_set_ici()
ici.SetPhiLimit(-30, 30, use_mirror=True)
output_workspaces = ici.WavRangeReduction(None, None, ici.DefaultTrans, combineDet="merged")
save_sector_reduction(output_workspaces, "_horizontal_sector")

ici.SetPhiLimit(60, 120, use_mirror=True)
output_workspaces=ici.WavRangeReduction(None, None, ici.DefaultTrans, combineDet="merged")
save_sector_reduction(output_workspaces, "_vertical_sector")
