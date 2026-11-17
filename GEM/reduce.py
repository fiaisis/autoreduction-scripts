from mantid.kernel import ConfigService
from pathlib import Path
from Diffraction.isis_powder.gem import Gem
from Diffraction.isis_powder import SampleDetails
import shutil
import uuid
import time
import yaml


######
# autoreduction
######

runno = "103802"
# Set the mode for reduction
mode = "PDF"
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

cycle = "cycle_26_3"

output = f"GEM{runno}.nxs"

offset_file = "GEM_Offsets_26_3_SRM_640_C.cal"
offset_file_base_path = Path(f"/extras/gem/{offset_file}")
rietveldvanrunnumbers = 96663
rietveldemptyrunnumbers = 96664
pdfvanrunnumbers = 97483
pdfemptyrunnumbers = 97484
first_cycle_run_no=int(runno) - 1

mapping_file_data = {
    f"{int(rietveldvanrunnumbers) - 1}-{runno}": {"label": f"{cycle}", "offset_file_name": f"{offset_file}",
                             "Rietveld": {"vanadium_run_numbers": f"{rietveldvanrunnumbers}",
                                    "empty_run_numbers": f"{rietveldemptyrunnumbers}"},
                              "PDF": {"vanadium_run_numbers": f"{pdfvanrunnumbers}",
                                    "empty_run_numbers": f"{pdfemptyrunnumbers}"}}
}

        
cal_mapping_file = f"GEM_{cycle}_calibration_mapping.yaml"
grouping_file = "GEM_Instrument_grouping.cal"
grouping_file_base_path = f"extras/gem/{grouping_file}"

# --- Persistent calibration directory (shared across runs on the Manila share) ---
persistent_calibration_directory = Path(r"/gem/Calibrations")
persistent_cycle_path = persistent_calibration_directory / cycle

# --- Per-run temporary calibration directory to avoid file locking between concurrent runs ---
temp_cal_dir = Path(f"/gem/tmp_cal_{runno}_{uuid.uuid4().hex[:8]}")
temp_cal_dir.mkdir(parents=True, exist_ok=True)

# --- Clean up orphaned temp directories from previous runs that did not clean up after themselves ---
MAX_TEMP_DIR_AGE_SECONDS = 6 * 60 * 60  # 6 hours (matches MAX_TIME_TO_COMPLETE_JOB)
gem_base = Path("/gem")
now = time.time()
for entry in gem_base.iterdir():
    if entry.is_dir() and entry.name.startswith("tmp_cal_"):
        try:
            dir_age = now - entry.stat().st_mtime
            if dir_age > MAX_TEMP_DIR_AGE_SECONDS:
                print(f"Cleaning up orphaned temp directory: {entry}")
                shutil.rmtree(entry, ignore_errors=True)
        except Exception as e:
            print(f"Warning: could not check/clean temp directory {entry}: {e}")


def generate_mapping_file(cal_mapping_file_path, mapping_file_data):
    try:
        print(f"creating mapping file with {mapping_file_data}:")
        with open(cal_mapping_file_path, 'w') as f:
            yaml.dump(mapping_file_data, f)
    except Exception as e:
        print(f"Error occurred while generating mapping file: {e}")


try:
    # Create cycle subdirectory inside the temp calibration directory
    temp_cycle_path = temp_cal_dir / cycle
    print(f"creating dir {temp_cycle_path}")
    temp_cycle_path.mkdir(parents=True, exist_ok=True)

    # Copy offset file from /extras into temp cycle directory
    print(f"Copying {offset_file} into {temp_cycle_path}")
    shutil.copy(offset_file_base_path, temp_cycle_path)

    # Copy grouping file from /extras into temp calibration root
    print(f"Copying {grouping_file} into {temp_cal_dir}")
    shutil.copy(grouping_file_base_path, temp_cal_dir)

    # Generate the mapping YAML inside the temp directory
    temp_mapping_file_path = temp_cal_dir / cal_mapping_file
    print(f"Generating mapping file {cal_mapping_file}")
    generate_mapping_file(temp_mapping_file_path, mapping_file_data)

    # Copy any existing splined vanadium files from the persistent calibration
    # directory so create_vanadium() can skip recomputation if already done
    persistent_cycle_path.mkdir(parents=True, exist_ok=True)
    for f in persistent_cycle_path.glob("*Splined*"):
        print(f"Reusing cached splined vanadium file: {f.name}")
        shutil.copy2(f, temp_cycle_path)

    gem = Gem(
        calibration_directory=temp_cal_dir,
        output_directory=Path("/output"),
        user_name="Autoreduction",
        mode = mode,
        vanadium_normalisation=van_norm,
        do_absorb_corrections=do_absorb_corrections,
        input_mode=input_mode,
        save_all=save_all,
    )

    # Vanadium only
    # isis_powder checks for existing splined vanadium files.
    # If they exist, create_vanadium is a no-op effectively.
    # If you pre-compute vanadium and store in /extras/gem/,
    # you can remove this block entirely.

    sample_details = SampleDetails(height=4.0, radius=0.2985, center=[0, 0, 0], shape='cylinder')
    sample_details.set_material(chemical_formula='Si', packing_fraction=0.6)
    sample_details.set_container(radius=0.3175, chemical_formula='V')
    gem.set_sample_details(sample=sample_details)

    gem.create_vanadium(
        first_cycle_run_no=first_cycle_run_no,
        calibration_mapping_file=temp_mapping_file_path,
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
        calibration_mapping_file=temp_mapping_file_path,
        do_absorb_corrections=do_absorb_corrections,
        multiple_scattering=False,
        input_mode=input_mode,
        mode=mode,
        run_number=runno,
        keep_raw_workspace=False,
        save_all=save_all,
        #focused_cropping_values=focused_cropping_values,
    )

    # Persist newly generated splined vanadium files back to the persistent
    # calibration directory so future runs can reuse them without recomputing
    for f in temp_cycle_path.glob("*Splined*"):
        dest = persistent_cycle_path / f.name
        if not dest.exists():
            print(f"Caching splined vanadium file for future runs: {f.name}")
            shutil.copy2(f, dest)

    print(f"Reduction completed for run {runno}. Output files: {output}")

finally:
    # Always clean up the per-run temp directory, even if the reduction failed
    print(f"Cleaning up temporary calibration directory: {temp_cal_dir}")
    shutil.rmtree(temp_cal_dir, ignore_errors=True)
