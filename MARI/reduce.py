from __future__ import print_function

import requests as requests
import os

output_dir = "/output"


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
        print(f"Attempting to get resource {url}")
        response = requests.get(url)
        if not response.ok:
            print(f"Failed to get resource from: {url}")
            print(f"Waiting {wait_time_seconds}...")
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


get_file_from_request("https://raw.githubusercontent.com/mantidproject/scriptrepository/master/direct_inelastic/"
                      "MARI/MARIReduction_Sample.py", "MARIReduction_Sample.py")
get_file_from_request("url_to_mask_file.xml", "mask_file.xml") # This url is inserted by IR-API transform
get_file_from_request("https://raw.githubusercontent.com/pace-neutrons/InstrumentFiles/"
                      "964733aec28b00b13f32fb61afa363a74dd62130/mari/mari_res2013.map", "mari_res2013.map")


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
