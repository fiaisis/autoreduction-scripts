from mantid.simpleapi import SANSLoad, SaveCanSAS1D, SANSSingleReduction, GroupWorkspaces, mtd, SaveNexusProcessed
from mantid import config
from sans.common.file_information import SANSFileInformationFactory
from sans.common.enums import ReductionMode
from sans.user_file.toml_parsers.toml_reader import TomlReader
from sans.user_file.toml_parsers.toml_v1_parser import TomlV1Parser
from sans.state.Serializer import Serializer

# Setup by rundetection
user_file = "/extras/loq/MaskFile.toml"
sample_scatter = "74044"  # Will need the 00 added for new cycles
cycle = "cycle_12_3"
sample_scatter_file_path = f"/archive/NDXLOQ/Instrument/data/{cycle}/LOQ{sample_scatter}.nxs"
sample_scatter_period = 1
sample_transmission = "74024"
sample_transmission_period = 1
sample_direct = "74014"
sample_direct_period = 1
can_scatter = "74019"
can_scatter_period = 1
can_transmission = "74020"
can_transmission_period = 1
can_direct = "74014"
can_direct_period = 1

output_name = "output_ws"
sample_thickness = 1.0
sample_geometry = "Disc"

# The following are defaults
instrument = "LOQ"

# Batch settings
multi_period = False

# Save and reduction options
reduction_type = "1D"
save_options = "memory"
save_type = "NxCanSAS"
zero_error_free = True
use_optimizations = True
reduction_mode = "Merged"
merge_fit_mode = "NoFit"
reduction_scale_factor = 1.0 
reduction_shift_factor = 0.0
radiation_source = "Spallation Neutron Source"
run_folder = f"run-{sample_scatter}"
output_path = f"/output/{run_folder}/"
config['defaultsave.directory'] = output_path


def generate_sans_state() -> str:
    read_toml = TomlReader.get_user_file_dict(user_file)
    file_info = SANSFileInformationFactory().create_sans_file_information(sample_scatter_file_path)
    all_states = TomlV1Parser(dict_to_parse=read_toml, file_information=file_info).get_all_states(file_info)
    all_states.data.sample_scatter = sample_scatter
    all_states.data.sample_scatter_period = sample_scatter_period
    all_states.data.sample_transmission = sample_transmission
    all_states.data.sample_transmission_period = sample_transmission_period
    all_states.data.sample_direct = sample_direct
    all_states.data.sample_direct_period = sample_direct_period
    all_states.data.can_scatter = can_scatter
    all_states.data.can_scatter_period = can_scatter_period
    all_states.data.can_transmission = can_transmission
    all_states.data.can_transmission_period = can_transmission_period
    all_states.data.can_direct = can_direct
    all_states.data.can_direct_period = can_direct_period
    all_states.data.sample_scatter_is_multi_period = multi_period
    all_states.reduction.reduction_mode = ReductionMode.MERGED
    return Serializer.to_json(all_states)


# Run workflow:
sans_state = generate_sans_state()
loaded_output = SANSLoad(SANSState=sans_state,
                         SampleScatterWorkspace=f'{sample_scatter}_sans_nxs',
                         SampleScatterMonitorWorkspace=f'{sample_scatter}_sans_nxs_monitors',
                         SampleTransmissionWorkspace=f'{sample_transmission}_trans_nxs',
                         SampleDirectWorkspace=f'{sample_direct}_direct_nxs',
                         CanScatterWorkspace=f'{can_scatter}_sans_nxs',
                         CanScatterMonitorWorkspace=f'{can_scatter}_sans_nxs_monitors',
                         CanTransmissionWorkspace=f'{can_transmission}_trans_nxs',
                         CanDirectWorkspace=f'{can_direct}_direct_nxs')

GroupWorkspaces(
    InputWorkspaces=f"{sample_scatter}_sans_nxs, {sample_scatter}_sans_nxs_monitors, {sample_transmission}_trans_nxs, {sample_direct}_direct_nxs, {can_scatter}_sans_nxs, {can_scatter}_sans_nxs_monitors, {can_transmission}_trans_nxs, {can_direct}_direct_nxs",
    OutputWorkspace="sans_raw_data")

# Output name definitions
output_lab_workspace_name = f"{instrument}-lab-{sample_scatter}"
output_hab_workspace_name = f"{instrument}-hab-{sample_scatter}"
output_merged_workspace_name = f"{instrument}-merged-{sample_scatter}"

SANSSingleReduction(SANSState=sans_state, 
                    SaveCan=True,
                    SampleScatterWorkspace=f'{sample_scatter}_sans_nxs',
                    SampleScatterMonitorWorkspace=f'{sample_scatter}_sans_nxs_monitors',
                    SampleTransmissionWorkspace=f'{sample_transmission}_trans_nxs',
                    SampleDirectWorkspace=f'{sample_direct}_direct_nxs',
                    CanScatterWorkspace=f'{can_scatter}_sans_nxs',
                    CanScatterMonitorWorkspace=f'{can_scatter}_sans_nxs_monitors',
                    CanTransmissionWorkspace=f'{can_transmission}_trans_nxs',
                    CanDirectWorkspace=f'{can_direct}_direct_nxs',
                    OutScaleFactor=reduction_scale_factor,
                    OutShiftFactor=reduction_shift_factor,
                    OutputWorkspaceLAB=output_lab_workspace_name,
                    OutputWorkspaceHAB=output_hab_workspace_name,
                    OutputWorkspaceHABScaled='output_hab_scaled',
                    OutputWorkspaceMerged=output_merged_workspace_name,
                    OutputWorkspaceLABCan='output_lab_can',
                    OutputWorkspaceHABCan='output_hab_can',
                    OutputWorkspaceLABSample='output_lab_sample',
                    OutputWorkspaceHABSample='output_hab_sample',
                    OutputWorkspaceCalculatedTransmission='output_calculated_trans',
                    OutputWorkspaceUnfittedTransmission='output_unfitted_trans',
                    OutputWorkspaceCalculatedTransmissionCan='output_calculated_trans_can',
                    OutputWorkspaceUnfittedTransmissionCan='output_unfitted_trans_can',
                    OutputWorkspaceLABCanNorm='output_lab_can_norm',
                    OutputWorkspaceLABCanCount='output_lab_can_count',
                    OutputWorkspaceHABCanCount='output_hab_can_count',
                    OutputWorkspaceHABCanNorm='output_hab_can_norm',
                    Version=1)


def save_workspace(workspace):
    save_file_name = f"{workspace.name()}.nxs"
    save_path = f"{output_path}/{save_file_name}"
    SaveNexusProcessed(workspace, save_path)


def save_cansas1d(workspace):
    save_file_path = f"{output_path}/{workspace.name()}.xml"
    SaveCanSAS1D(InputWorkspace=workspace,
                 Filename=save_file_path,
                 RadiationSource=radiation_source,
                 Geometry=sample_geometry,
                 Transmission='output_calculated_trans',
                 TransmissionCan='output_calculated_trans_can',
                 SampleTransmissionRunNumber=sample_transmission,
                 SampleDirectRunNumber=sample_direct,
                 CanScatterRunNumber=can_scatter,
                 CanDirectRunNumber=can_direct,
                 SampleThickness=sample_thickness)


output = []
output_workspaces = [mtd[output_lab_workspace_name], mtd[output_hab_workspace_name], mtd[output_merged_workspace_name]]
for workspace in output_workspaces:
    save_workspace(workspace)
    save_cansas1d(workspace)
    output.append(workspace.name)
