import sys
import os
import importlib

sys.path.append(os.path.dirname(__file__))

import reduction_utils

importlib.reload(reduction_utils)
import mantid
from mantid.simpleapi import mtd, Load, MergeMD, BinMD, CompactMD, SaveMD

# Default save directory (/output only for autoreduction as the RBNumber/autoreduced dir is mounted here)
output_dir = "/output"
mantid.config["defaultsave.directory"] = output_dir  # data_dir

# Starts live data collection:
# StartLiveData(Instrument='MERLIN_EVENT', Listener='ISISLiveEventDataListener', Address='NDXMERLIN:10000',
#    PreserveEvents=True, AccumulationMethod='Add', OutputWorkspace='lives')
# runno = 'lives'

# This could also be a workspace name as a string (e.g. for live data)
runno = "input"  # input loads the workspace  # Si data, use 69194 for quartz
wbvan = 69168
rotation_block_name = "Rot"
rotation_zero_angle = -2  # psi0 in Horace notation
rotation_bin_size = 5  # In degrees
lattice_pars = [5.43, 5.43, 5.43]  # In Angstrom
lattice_ang = [90, 90, 90]  # In degrees
uvector = "1, 0, 0"  # Reciprocal lattice vector parallel to incident beam direction when rotation=psi0
vvector = (
    "0, 1, 0"  # Reciprocal lattice vector perpendicular to u in the horizontal plane
)

wsname = runno if isinstance(runno, str) else f"MER{runno}"
if wsname not in mtd:
    ws = Load(wsname, OutputWorkspace=wsname)
else:
    ws = mtd[wsname]

eis = reduction_utils.autoei(ws)

for ei in eis:
    output_ws = reduction_utils.iliad(
        runno=wsname,
        wbvan=wbvan,
        ei=ei,
        Erange=[-0.5, 0.01, 0.85],
        cs_block=rotation_block_name,
        cs_bin_size=rotation_bin_size,
        cs_conv_to_md=True,
        cs_conv_pars={
            "lattice_pars": lattice_pars,
            "lattice_ang": lattice_ang,
            "u": uvector,
            "v": vvector,
            "psi0": rotation_zero_angle,
        },
        hard_mask_file="mask_25_1.xml",
        inst="merlin",
        powdermap="MERLIN_rings_251.xml",
    )
    allws = [w for w in mtd.getObjectNames() if w.startswith(f"{wsname}_{ei:g}meV")]
    wsout = MergeMD(",".join(allws), OutputWorkspace=f"MER{runno}_{ei:g}meV_1to1_md")
    mn = [wsout.getDimension(i).getMinimum() for i in range(4)]
    mx = [wsout.getDimension(i).getMaximum() for i in range(4)]
    wsbin = BinMD(
        wsout,
        AlignedDim0=f"[H,0,0],{mn[0]},{mx[0]},100",
        AlignedDim1=f"[0,K,0],{mn[1]},{mx[1]},100",
        AlignedDim2=f"[0,0,L],{mn[2]},{mx[2]},100",
        AlignedDim3=f"DeltaE,{mn[3]},{mx[3]},50",
    )
    wsbin = CompactMD(wsbin)
    SaveMD(
        wsbin,
        Filename=os.path.join(output_dir, f"MER{runno}_{ei:g}meV_1to1_mdhisto.nxs"),
    )
    mtd.remove(wsbin.name())
    mtd.remove(wsout.name())

# We do not want to remove the workspace
# for ws in allws:
#     mtd.remove(ws)
