# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import numpy as np
import requests as requests


def generate_input_path_for_run(run_number, cycle):
    return f"/archive/ndxosiris/Instrument/data/{cycle}/OSI{run_number}.nxs"

def get_file_from_request(url: str, path: str) -> None:
    """
    write the file from the url to the given path, retrying at most 3 times
    :param url: the url to get
    :param path: the path to write to
    :return: None
    """
    success = False
    attempts = 0
    wait_time_seconds = 15
    while attempts < 3:
        print(f"Attempting to get resource {url}", flush=True)
        response = requests.get(url)
        if not response.ok:
            print(f"Failed to get resource from: {url}", flush=True)
            print(f"Waiting {wait_time_seconds}...", flush=True)
            time.sleep(wait_time_seconds)
            attempts += 1
            wait_time_seconds *= 3
        else:
            with open(path, "w+") as fle:
                fle.write(response.text)
            success = True
            break

    if not success:
        raise RuntimeError(f"Reduction not possible with missing resource {url}")

# To change by automatic script
input_runs = ["108538", "108539"]
# This needs to be loaded from a shared repository of files
calibration_run_number = "00148587"
cycle = "cycle_14_1"
analyser = "graphite"
reflection = "002"
spectroscopy_reduction = True
diffraction_reduction = False

if not diffraction_reduction and not spectroscopy_reduction:
    raise RuntimeError("diffraction_reduction and spectroscopy_reduction are both false, so this will do nothing.")

# Defaults
instrument = "OSIRIS"
instrument_definition_directory = ConfigService.Instance().getString("instrumentDefinition.directory")
instrument_filename = instrument_definition_directory + instrument + "_Definition.xml"
instrument_workspace = LoadEmptyInstrument(Filename=instrument_filename, OutputWorkspace="instrument_workspace")
parameter_filename = instrument_definition_directory + instrument + "_" + analyser + "_" + reflection + "_Parameters.xml"
parameter_file = LoadParameterFile(Filename=parameter_filename, Workspace="instrument_workspace")
efixed = instrument_workspace.getInstrument().getComponentByName(analyser).getNumberParameter("Efixed")[0]
print(efixed)
spec_spectra_range = "963,1004"
diff_spectra_range = '3,962'
unit_x = "DeltaE"
fold_multiple_frames = False
diffraction_calibration_file = "osiris_041_RES10.cal"
get_file_from_request("https://raw.githubusercontent.com/fiaisis/autoreduction-scripts/cbf4e37365e112334bc7cee601553ebd0dbacc1d/OSIRIS/osiris_041_RES10.cal", diffraction_calibration_file)

# Generated
sum_runs = len(input_runs) > 1

input_file_paths = ""
for input_run in input_runs:
    input_file_paths += ", " + generate_input_path_for_run(input_run, cycle)
input_file_paths = input_file_paths[2:]  # Slice out the excess ", "
print(input_file_paths)
output_workspaces = []

# Generate calibration workspace

def generate_spec_calibration_workspace():
    # The following is already done above to generate the eFixed number:
    # instrument_definition_directory = ConfigService.Instance().getString("instrumentDefinition.directory")
    # instrument_filename = instrument_definition_directory + instrument + "_Definition.xml"
    # instrument_workspace = LoadEmptyInstrument(Filename=instrument_filename, OutputWorkspace="instrument_workspace")
    # parameter_filename = instrument_definition_directory + instrument + "_" + analyser + "_" + reflection + "_Parameters.xml"
    # parameter_file = LoadParameterFile(Filename=parameter_filename, Workspace="instrument_workspace")
    resolution = instrument_workspace.getInstrument().getComponentByName(analyser).getNumberParameter("resolution", True)[0]
    x = [-6 * resolution, -5 * resolution, -2 * resolution, 0, 2 * resolution]
    y = [1, 2, 3, 4]
    e = [0, 0, 0, 0]
    energy_workspace = CreateWorkspace(DataX=x, DataY=y, DataE=e, NSpec=1, UnitX="DeltaE", OutputWorkspace="energy_workspace")
    energy_workspace = ConvertToHistogram(InputWorkspace="energy_workspace", OutputWorkspace="energy_workspace")
    LoadInstrument(Workspace="energy_workspace", InstrumentName=instrument, RewriteSpectraMap=True)
    LoadParameterFile(Filename=parameter_filename, Workspace="energy_workspace")
    spectra_min = energy_workspace.getInstrument().getNumberParameter("spectra-min")[0]
    spectrum = energy_workspace.getSpectrum(0)
    spectrum.setSpectrumNo(int(spectra_min))
    spectrum.clearDetectorIDs()
    spectrum.addDetectorID(int(spectra_min))
    tof_workspace = ConvertUnits(InputWorkspace="energy_workspace", OutputWorkspace="tof_workspace", Target="TOF", EMode="Indirect", EFixed=efixed)
    tof_data = tof_workspace.readX(0)

    calibration_input_files = "OSIRIS" + calibration_run_number + ".nxs"
    peak_range = f"{tof_data[0]},{tof_data[2]}"
    background_range = f"{tof_data[3]},{tof_data[4]}"
    calibration_workspace_name = "osiris" + calibration_run_number + "_" + analyser + reflection + "_calib"
    return IndirectCalibration(InputFiles=calibration_input_files, DetectorRange=spec_spectra_range, PeakRange=peak_range, BackgroundRange=background_range, ScaleByFactor=0, ScaleFactor=1, LoadLogFiles=0, OutputWorkspace=calibration_workspace_name)

# Perform the reduction
if spectroscopy_reduction:
    calibration_workspace = generate_spec_calibration_workspace()
 
    output_workspace_prefix = instrument
    for input_run in input_runs:
        output_workspace_prefix += input_run + ","
    output_workspace_prefix = output_workspace_prefix[:-2] + f"_{analyser}_{reflection}_Reduced"  # Slice out the excess ", " and finalize prefix
    
    output_spec_ws_individual = ISISIndirectEnergyTransferWrapper(OutputWorkspace=output_workspace_prefix + "-individual", GroupingMethod="Individual", InputFiles=input_file_paths, SumFiles=sum_runs, CalibrationWorkspace=calibration_workspace, Instrument=instrument, Analyser=analyser, Reflection=reflection, EFixed=efixed, SpectraRange=spec_spectra_range, FoldMultipleFrames=fold_multiple_frames, UnitX=unit_x)

    output_spec_ws_all = ISISIndirectEnergyTransferWrapper(OutputWorkspace=output_workspace_prefix + "-all", GroupingMethod="All", InputFiles=input_file_paths, SumFiles=sum_runs, CalibrationWorkspace=calibration_workspace, Instrument=instrument, Analyser=analyser, Reflection=reflection, EFixed=efixed, SpectraRange=spec_spectra_range, FoldMultipleFrames=fold_multiple_frames, UnitX=unit_x)

    output_workspaces.append(output_spec_ws_individual)
    output_workspaces.append(output_spec_ws_all)

    # Also perform the diffspec reduction using the diffraction algorithm
    print("Reducing in diffspec mode using diffraction algorithm")
    output_diffspec_ws = ISISIndirectDiffractionReduction(InputFiles=input_file_paths, CalFile=diffraction_calibration_file, Instrument=instrument, SpectraRange=diff_spectra_range, OutputWorkspace=f"{instrument}{input_runs[0]}_diffspec_red")
    output_workspaces.append(output_diffspec_ws)
# 
# elif diffraction_reduction:
#     print("Producing a diffraction reduction")
#     for run_number in input_runs:
#         ws_name = instrument + run_number
#         drange_ws = OSIRISDiffractionReduction(Sample=ws_name, CalFile=calibration_file_path, OutputWorkspace=f'{ws_name}_dRange')
#         q_ws = ConvertUnits(InputWorkspace=drange_ws, OutputWorkspace=f'{ws_name}_q', Target='QSquared')
#         tof_ws = ConvertUnits(InputWorkspace=drange_ws, OutputWorkspace=f'{ws_name}_tof', Target='TOF')
#         output_workspaces.append(drange_ws)
#         output_workspaces.append(q_ws)
#         output_workspaces.append(tof_ws)
        
output = []
for workspace in output_workspaces:
    save_file_name = f"{workspace.name()}.nxs"
    save_path = f"/output/{save_file_name}"
    SaveNexusProcessed(workspace, save_path)
    output.append(save_file_name)
