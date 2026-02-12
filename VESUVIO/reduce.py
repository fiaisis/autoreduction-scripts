import requests
import re
import os
 
 
from mantid.simpleapi import (
    LoadVesuvio,
    CropWorkspace,
    Minus,
    Rebin,
    RebinToWorkspace,
    ISISIndirectDiffractionReduction,
    SaveNexusProcessed,
    EditInstrumentGeometry,
    ConvertUnits,
    ConvertToHistogram
)
from mantid import config
from mantid.api import AnalysisDataService
from VesuvioTransmission import VesuvioTransmission
 
 
# Define Utility functions
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
            print("Successfully obtained resource")
            break

    if not success:
        raise RuntimeError(f"Reduction not possible with missing resource {url}")


def run_alg(algorithm_class, args):
    """
    Run the algorithm more cleanly when imported outside of the simpleapi
    :param algorithm_class: A PythonAlgorythm class to be executed
    :param args: Arguments to pass as a dict to the properties of the algorithm
    :return: None
    """
    alg = algorithm_class()
    alg.initialize()
    for key, value in args.items():
        alg.setProperty(key, value)
    alg.execute()
 
# Get VesuvioTransmission
get_file_from_request(
    "https://raw.githubusercontent.com/fiaisis/autoreduction-scripts/2427463c9a0247b7d76e57493bb94b28b8a7f54b/VESUVIO/VesuvioTransmission.py",
    "VesuvioTransmission.py",
)
 
 
# Setup by rundetection
ip = "IP0005.par"
diff_ip = "IP0005.par"
empty_runs = "50309,50310,50311"
runno = "50312,50313,50314"
sum_runs = False
output_workspace_prefix = "vesuvio"
 
if "," in runno or "-" in runno:
    sum_runs = True

# Resolve file names and paths
file_name = runno
empty_runs = empty_runs
diffraction_input = runno

if sum_runs:
    if "," in runno:
        input_runs = runno.split(",")
    elif "-" in runno:
        input_runs = runno.split("-")



print(f"Starting with input: {file_name}")

if len(input_runs) > 6:
    output_workspace_prefix += str(input_runs[0]) + ","
    output_workspace_prefix += str(input_runs[1]) + ","
    output_workspace_prefix += str(input_runs[2])
    output_workspace_prefix += "..."
    output_workspace_prefix += str(input_runs[-3]) + ","
    output_workspace_prefix += str(input_runs[-2]) + ","
    output_workspace_prefix += str(input_runs[-1]) + ","
else:
    for input_run in input_runs:
        output_workspace_prefix += str(input_run) + ","
output_workspace_prefix = output_workspace_prefix[:-1] + f"_Reduced"  # Slice out the excess "," and finalize prefix


# Default constants
filepath_ip = f"/extras/vesuvio/{ip}"
diff_filepath_ip = f"/extras/vesuvio/{diff_ip}"
rebin_vesuvio_run_parameters = "50,1,500"
rebin_transmission_parameters = "0.6,-0.05,1.e7"
crop_min = 10
crop_max = 400
back_scattering_spectra = "3-134"
forward_scattering_spectra = "135-182"
cache_location = "/extras/vesuvio/cached_files/"

# Other configuration options
config["defaultsave.directory"] = "/output"
output = []
# Convert back scattering spectra to a value acceptable in ISISIndirectDiffractionReduction i.e. [3, 134] instead of "3-134":
back_scattering_spectra_range = []
back_scattering_spectra_range.extend(back_scattering_spectra.split("-"))
for index, value in enumerate(back_scattering_spectra_range):
    back_scattering_spectra_range[index] = int(value)

# Load Empty runs
LoadVesuvio(
    Filename=empty_runs,
    SpectrumList=back_scattering_spectra,
    Mode="SingleDifference",
    InstrumentParFile=filepath_ip,
    SumSpectra=True,
    OutputWorkspace="empty_back_sd",
)
LoadVesuvio(
    Filename=empty_runs,
    SpectrumList=back_scattering_spectra,
    Mode="DoubleDifference",
    InstrumentParFile=filepath_ip,
    SumSpectra=True,
    OutputWorkspace="empty_back_dd",
)
LoadVesuvio(
    Filename=empty_runs,
    SpectrumList=forward_scattering_spectra,
    Mode="FoilInOut",
    InstrumentParFile=filepath_ip,
    SumSpectra=True,
    OutputWorkspace="empty_gamma",
)
CropWorkspace(
    InputWorkspace="empty_gamma",
    XMin=crop_min,
    XMax=crop_max,
    OutputWorkspace="empty_gamma",
)

# Setup run file for processing, then process the file.
LoadVesuvio(
    Filename=file_name,
    SpectrumList=forward_scattering_spectra,
    Mode="SingleDifference",
    InstrumentParFile=filepath_ip,
    SumSpectra=True,
    OutputWorkspace=output_workspace_prefix + "_front",
)
LoadVesuvio(
    Filename=file_name,
    SpectrumList=back_scattering_spectra,
    Mode="SingleDifference",
    InstrumentParFile=filepath_ip,
    SumSpectra=True,
    OutputWorkspace=output_workspace_prefix +  "_back_sd",
)
ConvertToHistogram("empty_back_sd", OutputWorkspace="empty_back_sd")
ConvertToHistogram(output_workspace_prefix + "_back_sd", OutputWorkspace=output_workspace_prefix + "_back_sd")
RebinToWorkspace("empty_back_sd", output_workspace_prefix + "_back_sd", OutputWorkspace="empty_back_sd")
Minus(
    LHSWorkspace=output_workspace_prefix + "_back_sd",
    RHSWorkspace="empty_back_sd",
    OutputWorkspace=output_workspace_prefix +  "_back_sd",
)
LoadVesuvio(
    Filename=file_name,
    SpectrumList=back_scattering_spectra,
    Mode="DoubleDifference",
    InstrumentParFile=filepath_ip,
    SumSpectra=True,
    OutputWorkspace=output_workspace_prefix +  "_back_dd",
)
ConvertToHistogram("empty_back_dd", OutputWorkspace="empty_back_dd")
ConvertToHistogram(output_workspace_prefix + "_back_dd", OutputWorkspace=output_workspace_prefix +  "_back_dd")
RebinToWorkspace("empty_back_dd", output_workspace_prefix + "_back_dd", OutputWorkspace="empty_back_dd")
Minus(
    LHSWorkspace=output_workspace_prefix + "_back_dd",
    RHSWorkspace="empty_back_dd",
    OutputWorkspace=output_workspace_prefix +  "_back_dd",
)
Rebin(
    InputWorkspace=output_workspace_prefix + "_back_sd",
    OutputWorkspace=output_workspace_prefix + "_back_sd",
    Params=rebin_vesuvio_run_parameters,
)
Rebin(
    InputWorkspace=output_workspace_prefix + "_back_dd",
    OutputWorkspace=output_workspace_prefix + "_back_dd",
    Params=rebin_vesuvio_run_parameters,
)
Rebin(
    InputWorkspace=output_workspace_prefix + "_front",
    OutputWorkspace=output_workspace_prefix + "_front",
    Params=rebin_vesuvio_run_parameters,
)
 
# Save out LoadVesuvio results
SaveNexusProcessed(InputWorkspace=f"{output_workspace_prefix}_back_dd", Filename=f"{output_workspace_prefix}_back_dd.nxs")
output.append(f"{output_workspace_prefix}_back_dd.nxs")
SaveNexusProcessed(InputWorkspace=f"{output_workspace_prefix}_back_sd", Filename=f"{output_workspace_prefix}_back_sd.nxs")
output.append(f"{output_workspace_prefix}_back_sd.nxs")
SaveNexusProcessed(InputWorkspace=f"{output_workspace_prefix}_front", Filename=f"{output_workspace_prefix}_front.nxs")
output.append(f"{output_workspace_prefix}_front.nxs")
 
# Run diffraction
actual_diffraction_workspace = ISISIndirectDiffractionReduction(
    InputFiles=diffraction_input,
    OutputWorkspace=runno + "_diffraction",
    Instrument="VESUVIO",
    Mode="diffspec",
    SpectraRange=back_scattering_spectra_range,
    SumFiles=sum_runs,
    InstrumentParFile=diff_filepath_ip,
)

# Get the actual workspace name created since it differs from OutputWorkspace
diffraction_output = actual_diffraction_workspace.name()

SaveNexusProcessed(
    InputWorkspace=diffraction_output, Filename=f"{diffraction_output}.nxs"
)
output.append(f"{diffraction_output}.nxs")

# Run VesuvioTransmission
vesuvio_transmission_args = {
    "OutputWorkspace": runno,
    "Runs": file_name,
    "EmptyRuns": empty_runs,
    "Grouping": "SumOfAllRuns",
    "Target": "Energy",
    "Rebin": True,
    "RebinParameters": rebin_transmission_parameters,
    "CalculateXS": True,
}
run_alg(VesuvioTransmission, vesuvio_transmission_args)
transmission_output = runno + "_transmission"

 
SaveNexusProcessed(
    InputWorkspace=transmission_output, Filename=f"{transmission_output}.nxs"
)
output.append(f"{transmission_output}.nxs")
SaveNexusProcessed(
    InputWorkspace=f"{transmission_output}_XS", Filename=f"{transmission_output}_XS.nxs"
)
output.append(f"{transmission_output}_XS.nxs")

# Run LoadVesuvio for gamma
LoadVesuvio(
    Filename=file_name,
    SpectrumList="135-182",
    Mode="FoilInOut",
    InstrumentParFile=filepath_ip,
    SumSpectra=True,
    OutputWorkspace=output_workspace_prefix +  "_gamma",
)
CropWorkspace(
    InputWorkspace=output_workspace_prefix + "_gamma",
    XMin=crop_min,
    XMax=crop_max,
    OutputWorkspace=output_workspace_prefix +  "_gamma",
)
ConvertToHistogram("empty_gamma", OutputWorkspace="empty_gamma")
ConvertToHistogram(output_workspace_prefix + "_gamma", OutputWorkspace=output_workspace_prefix + "_gamma")
RebinToWorkspace("empty_gamma", output_workspace_prefix + "_gamma", OutputWorkspace="empty_gamma")
Minus(
    LHSWorkspace=output_workspace_prefix + "_gamma",
    RHSWorkspace="empty_gamma",
    OutputWorkspace=output_workspace_prefix +  "_gamma",
)
SaveNexusProcessed(InputWorkspace=f"{output_workspace_prefix}_gamma", Filename=f"{output_workspace_prefix}_gamma.nxs")
output.append(f"{output_workspace_prefix}_gamma.nxs")
 
EditInstrumentGeometry(
    Workspace=output_workspace_prefix + "_gamma",
    L2="0.0001",
    Polar="0",
    InstrumentName="VESUVIO_RESONANCE",
)
ConvertUnits(
    InputWorkspace=output_workspace_prefix + "_gamma", OutputWorkspace=output_workspace_prefix + "_gamma_E", Target="Energy"
)
SaveNexusProcessed(InputWorkspace=f"{output_workspace_prefix}_gamma_E", Filename=f"{output_workspace_prefix}_gamma_E.nxs")
output.append(f"{output_workspace_prefix}_gamma_E.nxs")
 