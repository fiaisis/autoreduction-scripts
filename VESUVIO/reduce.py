import requests
from mantid.simpleapi import LoadVesuvio, CropWorkspace, Minus, Rebin, ISISIndirectDiffractionReduction, SaveNexusProcessed, EditInstrumentGeometry, ConvertUnits
from mantid import config
import time
import re

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

def normalize_runs(run_input):
    """
    Convert run input (string or list) to a comma-separated string of run numbers.
    Handle ranges like '55956-55957' and lists like ['55956', '55957'].
    """
    if isinstance(run_input, list):
        return ",".join(map(str, run_input))
    
    # Remove whitespace and replace hyphens with commas for normalization if we want to treat everything as a list of runs
    # However, LoadVesuvio handles ranges. So we just need to ensure it's a string.
    # But for naming, we need the first and last.
    return str(run_input).replace(" ", "")

def get_run_base_name(run_input):
    """
    Determine the base name for workspaces and files.
    Single run: '54822'
    Multiple runs: '54822_54823'
    """
    normalized = normalize_runs(run_input)
    # Split by comma or hyphen to get run numbers
    run_parts = re.split(r'[,\-]', normalized)
    run_parts = [r for r in run_parts if r] # Remove empty strings
    
    if len(run_parts) == 0:
        return "unknown_run"
    
    if len(run_parts) == 1:
        return run_parts[0]
    
    # For multiple runs, use first and last
    return f"{run_parts[0]}_{run_parts[-1]}"

# Get VesuvioTransmission
get_file_from_request("https://raw.githubusercontent.com/fiaisis/autoreduction-scripts/2427463c9a0247b7d76e57493bb94b28b8a7f54b/VESUVIO/VesuvioTransmission.py", "VesuvioTransmission.py")
from VesuvioTransmission import VesuvioTransmission


# Setup by rundetection
ip = "IP0005.par"
empty_runs = "50309-50310"
runno = "52695-52696"

# Normalize run numbers and determine base name for workspaces/files
normalized_runno = normalize_runs(runno)
run_prefix = get_run_base_name(runno)

# Default constants
filepath_ip = f"/extras/vesuvio/{ip}"
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

# Load Empty runs
LoadVesuvio(Filename=empty_runs, SpectrumList=back_scattering_spectra, Mode="SingleDifference",
            InstrumentParFile=filepath_ip, SumSpectra=True, OutputWorkspace="empty_back_sd")
LoadVesuvio(Filename=empty_runs, SpectrumList=back_scattering_spectra, Mode="DoubleDifference",
            InstrumentParFile=filepath_ip, SumSpectra=True, OutputWorkspace="empty_back_dd")
LoadVesuvio(Filename=empty_runs, SpectrumList=forward_scattering_spectra, Mode="FoilInOut", InstrumentParFile=filepath_ip,
            SumSpectra=True, OutputWorkspace="empty_gamma")
CropWorkspace(InputWorkspace="empty_gamma", XMin=crop_min, XMax=crop_max, OutputWorkspace="empty_gamma")

# Setup run file for processing, then process the file.
LoadVesuvio(Filename=normalized_runno, SpectrumList=forward_scattering_spectra, Mode="SingleDifference", InstrumentParFile=filepath_ip, SumSpectra=True, OutputWorkspace=run_prefix+"_front")
LoadVesuvio(Filename=normalized_runno, SpectrumList=back_scattering_spectra, Mode="SingleDifference", InstrumentParFile=filepath_ip, SumSpectra=True, OutputWorkspace=run_prefix+"_back_sd")
Minus(LHSWorkspace=run_prefix+"_back_sd", RHSWorkspace="empty_back_sd", OutputWorkspace=run_prefix+"_back_sd")
LoadVesuvio(Filename=normalized_runno, SpectrumList=back_scattering_spectra, Mode="DoubleDifference", InstrumentParFile=filepath_ip, SumSpectra=True, OutputWorkspace=run_prefix+"_back_dd")
Minus(LHSWorkspace=run_prefix+"_back_dd", RHSWorkspace="empty_back_dd", OutputWorkspace=run_prefix+"_back_dd")
Rebin(InputWorkspace=run_prefix+"_back_sd", OutputWorkspace=run_prefix+"_back_sd", Params=rebin_vesuvio_run_parameters)
Rebin(InputWorkspace=run_prefix+"_back_dd", OutputWorkspace=run_prefix+"_back_dd", Params=rebin_vesuvio_run_parameters)
Rebin(InputWorkspace=run_prefix+"_front", OutputWorkspace=run_prefix+"_front", Params=rebin_vesuvio_run_parameters)

# Save out LoadVesuvio results
SaveNexusProcessed(InputWorkspace=f"{run_prefix}_back_dd", Filename=f"{run_prefix}_back_dd.nxs")
output.append(f"{run_prefix}_back_dd.nxs")
SaveNexusProcessed(InputWorkspace=f"{run_prefix}_back_sd", Filename=f"{run_prefix}_back_sd.nxs")
output.append(f"{run_prefix}_back_sd.nxs")
SaveNexusProcessed(InputWorkspace=f"{run_prefix}_front", Filename=f"{run_prefix}_front.nxs")
output.append(f"{run_prefix}_front.nxs")

# Run diffraction
ISISIndirectDiffractionReduction(InputFiles=normalized_runno,
                             OutputWorkspace=run_prefix+"_diffraction",
                             Instrument="VESUVIO",
                             Mode="diffspec",
                             SumFiles=True,
                             SpectraRange=back_scattering_spectra_range)
diffraction_output = "vesuvio" + run_prefix + "_diffspec_red"
SaveNexusProcessed(InputWorkspace=diffraction_output, Filename=f"{diffraction_output}.nxs")
output.append(f"{diffraction_output}.nxs")

# Run VesuvioTransmission
vesuvio_transmission_args = {
    "OutputWorkspace": run_prefix,
    "Runs": normalized_runno,
    "EmptyRuns": empty_runs,
    "Grouping": "SumOfAllRuns",
    "Target": "Energy",
    "Rebin": True,
    "RebinParameters": rebin_transmission_parameters,
    "CalculateXS": True
}
run_alg(VesuvioTransmission, vesuvio_transmission_args)
transmission_output = run_prefix + "_transmission"
SaveNexusProcessed(InputWorkspace=transmission_output, Filename=f"{transmission_output}.nxs")
output.append(f"{transmission_output}.nxs")
SaveNexusProcessed(InputWorkspace=f"{transmission_output}_XS", Filename=f"{transmission_output}_XS.nxs")
output.append(f"{transmission_output}_XS.nxs")

# Run LoadVesuvio for gamma
LoadVesuvio(Filename=normalized_runno, SpectrumList="135-182", Mode="FoilInOut", InstrumentParFile=filepath_ip, SumSpectra=True, OutputWorkspace=run_prefix+"_gamma")
CropWorkspace(InputWorkspace=run_prefix+"_gamma", XMin=crop_min, XMax=crop_max, OutputWorkspace=run_prefix+"_gamma")
Minus(LHSWorkspace=run_prefix + "_gamma", RHSWorkspace="empty_gamma", OutputWorkspace=run_prefix+"_gamma")
SaveNexusProcessed(InputWorkspace=f"{run_prefix}_gamma", Filename=f"{run_prefix}_gamma.nxs")
output.append(f"{run_prefix}_gamma.nxs")

EditInstrumentGeometry(Workspace=run_prefix+"_gamma", L2='0.0001', Polar='0', InstrumentName='VESUVIO_RESONANCE')
ConvertUnits(InputWorkspace=run_prefix+"_gamma", OutputWorkspace=run_prefix+"_gamma_E", Target='Energy')
SaveNexusProcessed(InputWorkspace=f"{run_prefix}_gamma_E", Filename=f"{run_prefix}_gamma_E.nxs")
output.append(f"{run_prefix}_gamma_E.nxs")
