from pathlib import Path
from isis_powder.gem import Gem
import os
import yaml


######
# autoreduction
######
cycle = "cycle_25_3"
offset_file = "offsets_2023_cycle231.cal"
rietveldvanrunnumbers = "96663"
rietveldemptyrunnumbers = "96664"
pdfvanrunnumbers = "97483"
pdfemptyrunnumbers = "97484"
rietveldpdfvanemptyrunnumbers = {"Rietveld": {"vanadium_run_numbers": f"{rietveldvanrunnumbers}",
                                              "empty_run_numbers": f"{rietveldemptyrunnumbers}"},
                                 "PDF": {"vanadium_run_numbers": f"{pdfvanrunnumbers}",
                                         "empty_run_numbers": f"{pdfemptyrunnumbers}"}}

runno = "97486"
# Set the mode for reduction
mode = "Rietveld"
# Set to False to skip vanadium normalisation step
van_norm = True
# Set to False to skip absorption corrections
do_absorb_corrections = True
# Indicates whether to account for the effects of multiple scattering when calculating 
# absorption corrections. If do_absorb_corrections is set to True this parameter must be set.
multiple_scattering = True
# Summed, Individual
input_mode = "Individual"
# Set to True to save all intermediate workspaces, False to only save final focused workspace
save_all = True

mapping_file_data = {
    f"{runno - 1}-{runno}": {"label": f"{cycle}", "offset_file_name": f"{offset_file}",
                             "Rietveld": {"vanadium_run_numbers": f"{rietveldvanrunnumbers}",
                                    "empty_run_numbers": f"{rietveldemptyrunnumbers}"},
                              "PDF": {"vanadium_run_numbers": f"{pdfvanrunnumbers}",
                                    "empty_run_numbers": f"{pdfemptyrunnumbers}"}}
}

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

cal_mapping_file = f"GEM_{cycle}_calibration_mapping.yaml"
calibration_directory = Path("Calibrations")
create_directory(calibration_directory)
cal_mapping_file_path = cycle / cal_mapping_file

output = "/output"
create_directory(Path(cycle))

def generate_mapping_file(cal_mapping_file_path, mapping_file_data):
    if not os.path.exists(cal_mapping_file_path):
        os.makedirs(os.path.dirname(cal_mapping_file_path), exist_ok=True)
    try:
        with open(cal_mapping_file_path, 'w') as f:
            yaml.safe_dump(mapping_file_data, f)
    except Exception as e:
        print(f"Error occurred while generating mapping file: {e}")

generate_mapping_file(cal_mapping_file_path, mapping_file_data)

gem = Gem(
    calibration_to_adjust=cal_mapping_file,
    calibration_directory=calibration_directory,
    output_directory=output,
    user_name="Autoreduction",
)

gem.create_cal(run_number=runno,
               calibration_mapping_file=cal_mapping_file
)

# Vanadium only
# isis_powder checks for existing splined vanadium files.
# If they exist, create_vanadium is a no-op effectively.
# If you pre-compute vanadium and store in /extras/gem/,
# you can remove this block entirely.

gem.create_vanadium(
    calibration_mapping_file=cal_mapping_file,
    mode=mode,
    do_absorb_corrections=do_absorb_corrections,
    multiple_scattering=multiple_scattering,
    spline_coefficient=120,
    texture_mode=False
)


# Focus
print(f"Starting focus for run {runno} with mode {mode} and input mode {input_mode}")

# Choice of cropping values for PDF or Rietveld mode
if mode == "Rietveld":
    focused_cropping_values = [
        (700, 19500),  # Bank 1
        (1000, 19500),  # Bank 2
        (1000, 19500),  # Bank 3
        (1000, 19500),  # Bank 4
        (1000, 18500),  # Bank 5
        (1000, 16750),  # Bank 6
    ]
elif mode == "PDF":
    focused_cropping_values = [
        (550, 19900),  # Bank 1
        (550, 19900),  # Bank 2
        (550, 19900),  # Bank 3
        (550, 19900),  # Bank 4
        (550, 18500),  # Bank 5
        (550, 16750),  # Bank 6
    ]
else:
    raise ValueError(f"Invalid mode: {mode}. Expected 'PDF' or 'Rietveld'.")

gem.focus(
    calibration_mapping_file=cal_mapping_file,
    do_absorb_corrections=do_absorb_corrections,
    input_mode=input_mode,
    mode=mode,
    run_number=runno,
    vanadium_normalisation=van_norm,
    unit_to_keep="dSpacing",
    keep_raw_workspace=False,
    save_all=save_all,
    focused_cropping_values=focused_cropping_values,
)

print(f"Reduction completed for run {runno}. Output files: {output}")
