from __future__ import print_function

import requests as requests
import os

output_dir = "/output"


def _get_sample_script() -> None:
    attempts = 0
    success = False
    wait_time = 5
    while attempts <= 3:
        with open("MARIReduction_Sample.py", "w+") as fle:
            response = requests.get("https://raw.githubusercontent.com/mantidproject/scriptrepository/master/direct_inelastic/MARI/MARIReduction_Sample.py")
            if not response.ok or "html" in response.text:
                print(f"Failed to get sample script, waiting {wait_time}seconds and trying again...")
                time.sleep(wait_time)
                wait_time *= 3
                attempts += 1
                continue
            text = response.text
            fle.write(text)
            success = True
            break
    if not success:
        print("Could not obtain the mari sample script, reduction is not possible")
        raise RuntimeError("Could not obtain the mari sample script, reduction is not possible")
  
with open("mask_file.xml", "w+") as fle:
    text = requests.get("url_to_mask_file.xml").text
    fle.write(text)
    

with open("mari_res2013.map", "w+") as fle:
    text = requests.get(
        "https://raw.githubusercontent.com/pace-neutrons/InstrumentFiles/964733aec28b00b13f32fb61afa363a74dd62130/mari/mari_res2013.map").text
    fle.write(text)


_get_sample_script()

def get_output_files():
    files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    return files


original_files = get_output_files()

from mantid import config
from MARIReduction_Sample import *
import time
import datetime
import sys

if sys.version_info > (3,):
    if sys.version_info < (3, 4):
        from imp import reload
    else:
        from importlib import reload
try:
    # Note: due to the mantid-python implementation, one needs to run this
    # script in Mantid  script window  TWICE!!!  to deploy the the changes made to MARIReduction_Sample.py file.
    sys.path.insert(0, output_dir)
    reload(sys.modules['MARIReduction_Sample'])
except:
    print("*** WARNING can not reload MARIReduction_Sample file")
    pass

# Run number and Ei
runno = 28581
sum_runs = False
ei = 'auto'

# White vanadium run number
wbvan = 28580

# Default save directory (/output only for autoreduction as the RBNumber/autoreduced dir is mounted here)
config['defaultsave.directory'] = output_dir  # data_dir

# Absolute normalisation parameters
# monovan=21803
# sam_mass=41.104
# sam_rmm=398.9439
monovan = 0
sam_mass = 0
sam_rmm = 0

# Set to true to remove the constant ToF background from the data.
remove_bkg = True

# If necessary, add any sequence of reduction paramerters defined in MARIParameters.xml file
# to the end ot the illiad string using the form: property=value
# (e.g.:  iliad_mari(runno,ei,wbvan,monovan,sam_mass,sam_rmm,sum_runs,check_background=False)
output_ws = iliad_mari(runno, ei, wbvan, monovan, sam_mass, sam_rmm, sum_runs, check_background=remove_bkg,
                       hard_mask_file='mask_file.xml')

# To run reduction _and_ compute density of states together uncomment this and comment iliad_mari above
# bkgruns and runno can be lists, which means those runs will be summed, and the sum is reduced
# bkgruns = 20941
# iliad_dos(runno, wbvan, ei, monovan, sam_mass, sam_rmm, sum_runs, background=bkgrun, temperature=5)

# Output set for autoreduction
now_files = get_output_files()
output = list(set(original_files) - set(now_files))
if not output:
    output = now_files
