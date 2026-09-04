from pathlib import Path
from isis_powder.gem import Gem


######
# autoreduction
######

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

cal_mapping_file = "calibration_mapping.yaml"
cwd = Path.cwd()
cal_mapping_file_path = Path(cwd) / cal_mapping_file

output = "/output"

gem = Gem(
    calibration_to_adjust=cal_mapping_file,
    calibration_directory=cwd, #find the calibration directory in the current working directory
    output_directory=cwd, #output files into the current working directory
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
