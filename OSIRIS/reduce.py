from mantid.simpleapi import *
import requests as requests


def generate_input_path_for_run(run_number, cycle):
    return f"/archive/ndxosiris/Instrument/data/{cycle}/OSI{run_number}.nxs"


# To change by automatic script
input_runs = [108538, 108539]
calibration_run_numbers = [149784, 149785, 149786]
cycle = "cycle_14_1"
analyser = "graphite"
reflection = "002"
spectroscopy_reduction = True
diffraction_reduction = False

if not diffraction_reduction and not spectroscopy_reduction:
    raise RuntimeError("diffraction_reduction and spectroscopy_reduction are both false, so this will do nothing.")

# Defaults and other generated inputs
instrument = "OSIRIS"
instrument_definition_directory = ConfigService.Instance().getString("instrumentDefinition.directory")
instrument_filename = instrument_definition_directory + instrument + "_Definition.xml"
instrument_workspace = LoadEmptyInstrument(Filename=instrument_filename, OutputWorkspace="instrument_workspace")
parameter_filename = instrument_definition_directory + instrument + "_" + analyser + "_" + reflection + "_Parameters.xml"
parameter_file = LoadParameterFile(Filename=parameter_filename, Workspace="instrument_workspace")
efixed = instrument_workspace.getInstrument().getComponentByName(analyser).getNumberParameter("Efixed")[0]
spec_spectra_range = "963,1004"
diff_spectra_range = '3,962'
unit_x = "DeltaE"
fold_multiple_frames = False
diffraction_calibration_file = "/extras/osiris/osiris_041_RES10.cal"

sum_runs = len(input_runs) > 1

input_file_paths = ""
for input_run in input_runs:
    input_file_paths += ", " + generate_input_path_for_run(input_run, cycle)
input_file_paths = input_file_paths[2:]  # Slice out the excess ", "
print(input_file_paths)
output = []


# Generate calibration workspace function
def generate_spec_calibration_workspace():
    resolution = \
    instrument_workspace.getInstrument().getComponentByName(analyser).getNumberParameter("resolution", True)[0]
    x = [-6 * resolution, -5 * resolution, -2 * resolution, 0, 2 * resolution]
    y = [1, 2, 3, 4]
    e = [0, 0, 0, 0]
    CreateWorkspace(DataX=x, DataY=y, DataE=e, NSpec=1, UnitX="DeltaE", OutputWorkspace="energy_workspace")
    energy_workspace = ConvertToHistogram(InputWorkspace="energy_workspace", OutputWorkspace="energy_workspace")
    LoadInstrument(Workspace="energy_workspace", InstrumentName=instrument, RewriteSpectraMap=True)
    LoadParameterFile(Filename=parameter_filename, Workspace="energy_workspace")
    spectra_min = energy_workspace.getInstrument().getNumberParameter("spectra-min")[0]
    spectrum = energy_workspace.getSpectrum(0)
    spectrum.setSpectrumNo(int(spectra_min))
    spectrum.clearDetectorIDs()
    spectrum.addDetectorID(int(spectra_min))
    tof_workspace = ConvertUnits(InputWorkspace="energy_workspace", OutputWorkspace="[]tof_workspace", Target="TOF",
                                 EMode="Indirect", EFixed=efixed)
    tof_data = tof_workspace.readX(0)

    calibration_input_files = ""
    for calibration_run_number in calibration_run_numbers:
        calibration_input_files += "OSIRIS" + str(calibration_run_number) + ".nxs" + ","
    calibration_input_files = calibration_input_files[:-1]
    peak_range = f"{tof_data[0]},{tof_data[2]}"
    background_range = f"{tof_data[3]},{tof_data[4]}"
    calibration_workspace_name = instrument.lower() + str(
        calibration_run_numbers[0]) + "_" + analyser + reflection + "_calib"
    return IndirectCalibration(InputFiles=calibration_input_files, DetectorRange=spec_spectra_range,
                               PeakRange=peak_range, BackgroundRange=background_range, ScaleByFactor=0, ScaleFactor=1,
                               LoadLogFiles=0, OutputWorkspace=calibration_workspace_name)


def save_workspace(workspace):
    save_file_name = f"{workspace.name()}.nxs"
    save_path = f"/output/{save_file_name}"
    SaveNexusProcessed(workspace, save_path)
    output.append(save_file_name)


# Perform the reduction
if spectroscopy_reduction:
    calibration_workspace = generate_spec_calibration_workspace()
    output_workspace_prefix = instrument
    if len(input_runs) > 6:
        output_workspace_prefix += str(input_runs[0]) + ","
        output_workspace_prefix += str(input_runs[1]) + ","
        output_workspace_prefix += str(input_runs[2]) + ","
        output_workspace_prefix += "..."
        output_workspace_prefix += str(input_runs[-3]) + ","
        output_workspace_prefix += str(input_runs[-2]) + ","
        output_workspace_prefix += str(input_runs[-1]) + ","
    else:
        for input_run in input_runs:
            output_workspace_prefix += str(input_run) + ","
    output_workspace_prefix = output_workspace_prefix[
                              :-1] + f"_{analyser}_{reflection}_Reduced"  # Slice out the excess "," and finalize prefix

    output_spec_ws_individual = ISISIndirectEnergyTransferWrapper(
        OutputWorkspace=output_workspace_prefix + "-individual", GroupingMethod="Individual",
        InputFiles=input_file_paths, SumFiles=sum_runs, CalibrationWorkspace=calibration_workspace,
        Instrument=instrument, Analyser=analyser, Reflection=reflection, EFixed=efixed, SpectraRange=spec_spectra_range,
        FoldMultipleFrames=fold_multiple_frames, UnitX=unit_x)
    save_workspace(output_spec_ws_individual)

    output_spec_ws_all = ISISIndirectEnergyTransferWrapper(OutputWorkspace=output_workspace_prefix + "-all",
                                                           GroupingMethod="All", InputFiles=input_file_paths,
                                                           SumFiles=sum_runs,
                                                           CalibrationWorkspace=calibration_workspace,
                                                           Instrument=instrument, Analyser=analyser,
                                                           Reflection=reflection, EFixed=efixed,
                                                           SpectraRange=spec_spectra_range,
                                                           FoldMultipleFrames=fold_multiple_frames, UnitX=unit_x)
    save_workspace(output_spec_ws_all)

    # Also perform the diffspec reduction using the diffraction algorithm
    output_diffspec_ws = ISISIndirectDiffractionReduction(InputFiles=input_file_paths,
                                                          CalFile=diffraction_calibration_file, Instrument=instrument,
                                                          SpectraRange=diff_spectra_range,
                                                          OutputWorkspace=f"{instrument}{input_runs[0]}_diffspec_red")
    save_workspace(output_diffspec_ws)
