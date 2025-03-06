import requests
import os

from pathlib import Path

from mantid.simpleapi import LoadVesuvio, CropWorkspace, Minus, Rebin, ISISIndirectDiffractionReduction, SaveNexusProcessed, LoadNexusProcessed
from mantid import config

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
get_file_from_request("https://raw.githubusercontent.com/fiaisis/autoreduction-scripts/2427463c9a0247b7d76e57493bb94b28b8a7f54b/VESUVIO/VesuvioTransmission.py", "VesuvioTransmission.py")
from VesuvioTransmission import VesuvioTransmission


# Setup by rundetection
ip="/extras/vesuvio/IP0005.par"
empty_runs = "50309-50341"
runno = "52695"

# Default constants
use_cache = True
rebin_vesuvio_run_parameters = "50,1,500"
rebin_transmission_parameters="0.6,-0.05,1.e7"
crop_min = 10
crop_max = 400
back_scattering_spectra = "3-134"
forward_scattering_spectra = "135-182"
cache_location="/extras/vesuvio/cached_files/"

# Other configuration options
config['defaultsave.directory'] = "/output"
output = []

# Convert back scattering spectra to a value acceptable in ISISIndirectDiffractionReduction i.e. [3, 134] instead of "3-134":
back_scattering_spectra_range = []
back_scattering_spectra_range.extend(back_scattering_spectra.split("-"))
for index, value in enumerate(back_scattering_spectra_range):
    back_scattering_spectra_range[index] = int(value)


# Load Empty runs if not cached
def load_empties():
    LoadVesuvio(Filename=empty_runs, SpectrumList=back_scattering_spectra, Mode="SingleDifference",
                InstrumentParFile=ip, SumSpectra=True, OutputWorkspace="empty_back_sd")
    LoadVesuvio(Filename=empty_runs, SpectrumList=back_scattering_spectra, Mode="DoubleDifference",
                InstrumentParFile=ip, SumSpectra=True, OutputWorkspace="empty_back_dd")
    LoadVesuvio(Filename=empty_runs, SpectrumList=forward_scattering_spectra, Mode="FoilInOut", InstrumentParFile=ip,
                SumSpectra=True, OutputWorkspace="empty_gamma")


def load_and_cache_file(filepath, spectrum_list, mode, outputworkspace):
    LoadVesuvio(Filename=empty_runs, SpectrumList=spectrum_list, Mode=mode, InstrumentParFile=ip, SumSpectra=True, OutputWorkspace=outputworkspace)
    SaveNexusProcessed(Filename=filepath, InputWorkspace=outputworkspace)


def load_vesuvio_from_cache_if_possible(filename, spectrum_list, mode, outputworkspace):
    generated_filename = filename + "_" + mode + "_" + spectrum_list + "_" + Path(ip).stem
    filepath = cache_location + generated_filename + ".nxs"
    if os.path.exists(filepath):
        print(f"Cache hit, found file: {filepath}")
        LoadNexusProcessed(Filename=filepath, OutputWorkspace=outputworkspace)
    else:
        print(f"Cache miss, could not find file: {filepath}")
        load_and_cache_file(filepath, spectrum_list, mode, outputworkspace)


if not use_cache:
    load_empties()
else:
    load_vesuvio_from_cache_if_possible(runno + "_empty_back", back_scattering_spectra, "SingleDifference", "empty_back_sd")
    load_vesuvio_from_cache_if_possible(runno + "_empty_back", back_scattering_spectra, "DoubleDifference", "empty_back_dd")
    load_vesuvio_from_cache_if_possible(runno + "_empty_gamma", back_scattering_spectra, "FoilInOut", "empty_gamma")

CropWorkspace(InputWorkspace="empty_gamma", XMin=crop_min, XMax=crop_max, OutputWorkspace="empty_gamma")

# Setup run file for processing, then process the file.
LoadVesuvio(Filename=runno, SpectrumList=forward_scattering_spectra, Mode="SingleDifference", InstrumentParFile=ip, SumSpectra=True, OutputWorkspace=runno+"_front")
LoadVesuvio(Filename=runno, SpectrumList=back_scattering_spectra, Mode="SingleDifference", InstrumentParFile=ip, SumSpectra=True, OutputWorkspace=runno+"_back_sd")
Minus(LHSWorkspace=runno+"_back_sd", RHSWorkspace="empty_back_sd", OutputWorkspace=runno+"_back_sd")
LoadVesuvio(Filename=runno, SpectrumList=back_scattering_spectra, Mode="DoubleDifference", InstrumentParFile=ip, SumSpectra=True, OutputWorkspace=runno+"_back_dd")
Minus(LHSWorkspace=runno+"_back_dd", RHSWorkspace="empty_back_dd", OutputWorkspace=runno+"_back_dd")
Rebin(InputWorkspace=runno+"_back_sd", OutputWorkspace=runno+"_back_sd", Params=rebin_vesuvio_run_parameters)
Rebin(InputWorkspace=runno+"_back_dd", OutputWorkspace=runno+"_back_dd", Params=rebin_vesuvio_run_parameters)
Rebin(InputWorkspace=runno+"_front", OutputWorkspace=runno+"_front", Params=rebin_vesuvio_run_parameters)

# Save out LoadVesuvio results
SaveNexusProcessed(InputWorkspace=f"{runno}_back_dd", Filename=f"{runno}_back_dd.nxs")
output.append(f"{runno}_back_dd.nxs")
SaveNexusProcessed(InputWorkspace=f"{runno}_back_sd", Filename=f"{runno}_back_sd.nxs")
output.append(f"{runno}_back_sd.nxs")
SaveNexusProcessed(InputWorkspace=f"{runno}_front", Filename=f"{runno}_front.nxs")
output.append(f"{runno}_back_front.nxs")

# Run diffraction
ISISIndirectDiffractionReduction(InputFiles=runno,
                             OutputWorkspace=runno+"_diffraction",
                             Instrument="VESUVIO",
                             Mode="diffspec",
                             SpectraRange=back_scattering_spectra_range)
diffraction_output = "vesuvio" + runno + "diffspec_red"
SaveNexusProcessed(InputWorkspace=diffraction_output, Filename=f"{diffraction_output}.nxs")
output.append(f"{diffraction_output}.nxs")

# Run VesuvioTransmission
vesuvio_transmission_args = {
    "OutputWorkspace": runno,
    "Runs": runno,
    "EmptyRuns": empty_runs,
    "Grouping": "SumOfAllRuns",
    "Target": "Energy",
    "Rebin": True,
    "RebinParameters": rebin_transmission_parameters,
    "CalculateXS": True
}
run_alg(VesuvioTransmission, vesuvio_transmission_args)
transmission_output = runno + "_transmission"
SaveNexusProcessed(InpurWorkspace=transmission_output, Filename=f"{transmission_output}.nxs")
output.append(f"{transmission_output}.nxs")
SaveNexusProcessed(InpurWorkspace=f"{transmission_output}_XS", Filename=f"{transmission_output}_XS.nxs")
output.append(f"{transmission_output}_XS.nxs")

# Run LoadVesuvio for gamma
LoadVesuvio(Filename=runno, SpectrumList="135-182", Mode="FoilInOut", InstrumentParFile=ip, SumSpectra=True, OutputWorkspace=runno+"_gamma")
CropWorkspace(InputWorkspace=runno+"_gamma", XMin=crop_min, XMax=crop_max, OutputWorkspace=runno+"_gamma")
Minus(LHSWorkspace=runno + "_gamma", RHSWorkspace="empty_gamma", OutputWorkspace=runno+"_gamma")
SaveNexusProcessed(InputWorkspace=f"{runno}_gamma", Filename=f"{runno}_gamma.nxs")
output.append(f"{runno}_gamma.nxs")
