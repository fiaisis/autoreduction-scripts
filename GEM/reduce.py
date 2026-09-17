from pathlib import Path
from isis_powder.gem import Gem
from isis_powder import SampleDetails
import shutil
import yaml


######
# autoreduction
######

runno = "96789"
# Set the mode for reduction
mode = "Rietveld"
# Set to False to skip vanadium normalisation step
van_norm = True
# Set to False to skip absorption corrections
do_absorb_corrections = False
# Indicates whether to account for the effects of multiple scattering when calculating 
# absorption corrections. If do_absorb_corrections is set to True this parameter must be set.
multiple_scattering = False
# Summed, Individual
input_mode = "Individual"
# Set to True to save all intermediate workspaces, False to only save final focused workspace
save_all = True

cycle = "cycle_25_1"


offset_file = "offsets_2023_cycle231.cal"
offset_file_base_path = Path("/extras/gem/{offset_file}")
rietveldvanrunnumbers = "96663"
rietveldemptyrunnumbers = "96664"
pdfvanrunnumbers = "97483"
pdfemptyrunnumbers = "97484"
first_cycle_run_no=int(runno) - 1

mapping_file_data = {
    f"{int(rietveldvanrunnumbers) - 1}-{runno}": {"label": f"{cycle}", "offset_file_name": f"{offset_file}",
                             "Rietveld": {"vanadium_run_numbers": f"{rietveldvanrunnumbers}",
                                    "empty_run_numbers": f"{rietveldemptyrunnumbers}"},
                              "PDF": {"vanadium_run_numbers": f"{pdfvanrunnumbers}",
                                    "empty_run_numbers": f"{pdfemptyrunnumbers}"}}
}

def generate_path(path: Path):
    if path.exists:
        print(f"Path {path} already exists")
        return
    else:
        return path.mkdir(parents=True)

        
cal_mapping_file = f"GEM_{cycle}_calibration_mapping.yaml"
calibration_directory = Path(r"/extras/gem/Calibrations")
cal_cycle_path = Path(calibration_directory, cycle)
print(f"creating dir {cal_cycle_path}")
generate_path(cal_cycle_path)
print(f"Moving {offset_file} into {cal_cycle_path}")
shutil.copy(offset_file_base_path, cal_cycle_path)

cal_mapping_file_path = Path(calibration_directory, cal_mapping_file)
offset_file_path = Path(calibration_directory, offset_file)

def generate_mapping_file(cal_mapping_file_path, mapping_file_data):
    try:
        print(f"creating mapping file with {mapping_file_data}:")
        with open(cal_mapping_file_path, 'w') as f:
            yaml.dump(mapping_file_data, f)
    except Exception as e:
        print(f"Error occurred while generating mapping file: {e}")

print(f"Generating mapping file {cal_mapping_file}")
generate_mapping_file(cal_mapping_file_path, mapping_file_data)


output = "/output"

gem = Gem(
    calibration_directory=calibration_directory,
    output_directory=output,
    user_name="Autoreduction",
    mode = mode,
    vanadium_normalisation=van_norm,
    do_absorb_corrections=do_absorb_corrections,
    input_mode=input_mode,
    save_all=save_all,
    mayers_mult_scat_events=100 #reduce points for monte carlo sim for testing, remove for prod
)

# Vanadium only
# isis_powder checks for existing splined vanadium files.
# If they exist, create_vanadium is a no-op effectively.
# If you pre-compute vanadium and store in /extras/gem/,
# you can remove this block entirely.

#sample_details = SampleDetails(height=4.0, radius=0.2985, center=[0, 0, 0], shape='cylinder')
#sample_details.set_material(chemical_formula='Si', packing_fraction=0.6)
#sample_details.set_container(radius=0.3175, chemical_formula='V')
#gem.set_sample_details(sample=sample_details)

gem.create_vanadium(
    first_cycle_run_no=first_cycle_run_no,
    calibration_mapping_file=cal_mapping_file_path,
    mode=mode,
    do_absorb_corrections=True, #these need to be true for vanadium
    multiple_scattering=True, #Might need to be true always
    spline_coefficient=30,
    texture_mode=False
)


# Focus
print(f"Starting focus for run {runno} with mode {mode} and input mode {input_mode}")

# Choice of cropping values for PDF or Rietveld mode
# if mode == "Rietveld":
#     focused_cropping_values = [
#         (700, 19500),  # Bank 1
#         (1000, 19500),  # Bank 2
#         (1000, 19500),  # Bank 3
#         (1000, 19500),  # Bank 4
#         (1000, 18500),  # Bank 5
#         (1000, 16750),  # Bank 6
#     ]
# elif mode == "PDF":
#     focused_cropping_values = [
#         (550, 19900),  # Bank 1
#         (550, 19900),  # Bank 2
#         (550, 19900),  # Bank 3
#         (550, 19900),  # Bank 4
#         (550, 18500),  # Bank 5
#         (550, 16750),  # Bank 6
#     ]
# else:
#     raise ValueError(f"Invalid mode: {mode}. Expected 'PDF' or 'Rietveld'.")

gem.focus(
    calibration_mapping_file=cal_mapping_file_path,
    do_absorb_corrections=do_absorb_corrections,
    multiple_scattering=False,
    input_mode=input_mode,
    mode=mode,
    run_number=runno,
    unit_to_keep="dSpacing",
    keep_raw_workspace=False,
    save_all=save_all,
    #focused_cropping_values=focused_cropping_values,
)

print(f"Reduction completed for run {runno}. Output files: {output}")
