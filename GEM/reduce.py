from pathlib import Path
from isis_powder.gem import Gem

runno = "97486"
mode = "Rietveld"  # PDF, Rietveld
input_mode = "Individual"  # Summed, Individual
vanadium_runno = '97482'
van_norm = True  # Set to False to skip vanadium normalisation step
save_all = False  # Set to True to save all intermediate workspaces, False to only save final focused workspace

calibration_dir = '/extras/gem/calibration_files'
splined_vanadium_dir = '/extras/gem/splined_vanadium'
config_file = '/extras/gem/Gem_config_example_25_3.yaml'
output_dir = Path('/output')
output = []

gem = Gem(
    user_name="Autoreduction",
    config_file=config_file,
)

# Vanadium only
# isis_powder checks for existing splined vanadium files.
# If they exist, create_vanadium is a no-op effectively.
# If you pre-compute vanadium and store in /extras/gem/, 
# you can remove this block entirely.
try:
    gem.create_vanadium(
        first_cycle_run_no=vanadium_runno,
        mode=mode,
        do_absorb_corrections=True,
        multiple_scattering=False,
        spline_coefficient=120,
    )
    print("Vanadium created successfully")
except Exception as e:
    print(f"Error occurred while creating vanadium: {e}")
    print("Attempting to continue with existing vanadium file if available")

# Focus
print(f"Starting focus for run {runno} with mode {mode} and input mode {input_mode}")
    
#Choice of cropping values for PDF or Rietveld mode
if mode == "Rietveld":
    focused_cropping_values = [(700, 19500),  # Bank 1
                                (1000, 19500),  # Bank 2
                                (1000, 19500),  # Bank 3
                                (1000, 19500),  # Bank 4
                                (1000, 18500),  # Bank 5
                                (1000, 16750)   # Bank 6
                                ]
elif mode == "PDF":
    focused_cropping_values = [(550, 19900),  # Bank 1
                                (550, 19900),  # Bank 2
                                (550, 19900),  # Bank 3
                                (550, 19900),  # Bank 4
                                (550, 18500),  # Bank 5
                                (550, 16750)   # Bank 6
                                ]
else:
    raise ValueError(f"Invalid mode: {mode}. Expected 'PDF' or 'Rietveld'.")

focused = gem.focus(
        run_number=runno,
        unit_to_keep="dSpacing",
        mode=mode,
        input_mode=input_mode,
        keep_raw_workspace=False,
        save_all=save_all,
        focused_cropping_values=focused_cropping_values,
        vanadium_normalisation=van_norm,
)

# Collect output files
output_path = Path(output_dir)
for path in output_path.rglob("*"):
    if path.is_file():
        output.append(str(path.name))
        print(f"Output file: {path.name}")

print(f"Reduction completed for run {runno}. Output files: {output}")
