import requests


from mantid.simpleapi import (
    LoadVesuvio,
    CropWorkspace,
    Minus,
    Rebin,
    ISISIndirectDiffractionReduction,
    SaveNexusProcessed,
    EditInstrumentGeometry,
    ConvertUnits,
)
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
get_file_from_request(
    "https://raw.githubusercontent.com/fiaisis/autoreduction-scripts/2427463c9a0247b7d76e57493bb94b28b8a7f54b/VESUVIO/VesuvioTransmission.py",
    "VesuvioTransmission.py",
)
from VesuvioTransmission import VesuvioTransmission


# Setup by rundetection
ip = "IP0005.par"
empty_runs = "50309-50341"
runno = "52695"
file_name = (
    requests.get(
        f"http://data.isis.rl.ac.uk/where.py/unixdir?name=VESUVIO{runno}"
    ).text.strip("\n")
    + f"/VESUVIO000{runno}.raw"
)

print(f"found filepath: {file_name}")

# Default constants
filepath_ip = f"/extras/vesuvio/{ip}"
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
    OutputWorkspace=runno + "_front",
)
LoadVesuvio(
    Filename=file_name,
    SpectrumList=back_scattering_spectra,
    Mode="SingleDifference",
    InstrumentParFile=filepath_ip,
    SumSpectra=True,
    OutputWorkspace=runno + "_back_sd",
)
Minus(
    LHSWorkspace=runno + "_back_sd",
    RHSWorkspace="empty_back_sd",
    OutputWorkspace=runno + "_back_sd",
)
LoadVesuvio(
    Filename=file_name,
    SpectrumList=back_scattering_spectra,
    Mode="DoubleDifference",
    InstrumentParFile=filepath_ip,
    SumSpectra=True,
    OutputWorkspace=runno + "_back_dd",
)
Minus(
    LHSWorkspace=runno + "_back_dd",
    RHSWorkspace="empty_back_dd",
    OutputWorkspace=runno + "_back_dd",
)
Rebin(
    InputWorkspace=runno + "_back_sd",
    OutputWorkspace=runno + "_back_sd",
    Params=rebin_vesuvio_run_parameters,
)
Rebin(
    InputWorkspace=runno + "_back_dd",
    OutputWorkspace=runno + "_back_dd",
    Params=rebin_vesuvio_run_parameters,
)
Rebin(
    InputWorkspace=runno + "_front",
    OutputWorkspace=runno + "_front",
    Params=rebin_vesuvio_run_parameters,
)

# Save out LoadVesuvio results
SaveNexusProcessed(InputWorkspace=f"{runno}_back_dd", Filename=f"{runno}_back_dd.nxs")
output.append(f"{runno}_back_dd.nxs")
SaveNexusProcessed(InputWorkspace=f"{runno}_back_sd", Filename=f"{runno}_back_sd.nxs")
output.append(f"{runno}_back_sd.nxs")
SaveNexusProcessed(InputWorkspace=f"{runno}_front", Filename=f"{runno}_front.nxs")
output.append(f"{runno}_back_front.nxs")

# Run diffraction
ISISIndirectDiffractionReduction(
    InputFiles=file_name,
    OutputWorkspace=runno + "_diffraction",
    Instrument="VESUVIO",
    Mode="diffspec",
    SpectraRange=back_scattering_spectra_range,
)
diffraction_output = "vesuvio" + runno + "_diffspec_red"
SaveNexusProcessed(
    InputWorkspace=diffraction_output, Filename=f"{diffraction_output}.nxs"
)
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
    OutputWorkspace=runno + "_gamma",
)
CropWorkspace(
    InputWorkspace=runno + "_gamma",
    XMin=crop_min,
    XMax=crop_max,
    OutputWorkspace=runno + "_gamma",
)
Minus(
    LHSWorkspace=runno + "_gamma",
    RHSWorkspace="empty_gamma",
    OutputWorkspace=runno + "_gamma",
)
SaveNexusProcessed(InputWorkspace=f"{runno}_gamma", Filename=f"{runno}_gamma.nxs")
output.append(f"{runno}_gamma.nxs")

EditInstrumentGeometry(
    Workspace=runno + "_gamma",
    L2="0.0001",
    Polar="0",
    InstrumentName="VESUVIO_RESONANCE",
)
ConvertUnits(
    InputWorkspace=runno + "_gamma", OutputWorkspace=runno + "_gamma_E", Target="Energy"
)
SaveNexusProcessed(InputWorkspace=f"{runno}_gamma_E", Filename=f"{runno}_gamma_E.nxs")
output.append(f"{runno}_gamma_E.nxs")
