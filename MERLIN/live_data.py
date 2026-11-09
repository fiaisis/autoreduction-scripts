# BEGIN FIA BLOCK

import os
import sys
import time
from datetime import datetime
from epics import caget


import requests

sys.path.append(os.path.dirname(__file__))
    
def execute():    

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
    
    
    git_sha = "1408532f20bd9113e91a107bebdef7a45c5fd903"
    print(f"mantidproject/direct_reduction sha is {git_sha}")
    # Only needed for fixes with regards to reductions during MARI issues
    get_file_from_request(
        f"https://raw.githubusercontent.com/mantidproject/direct_reduction/{git_sha}/reduction_files/reduction_utils.py",
        "reduction_utils.py",
    )
    get_file_from_request(
        f"https://raw.githubusercontent.com/mantidproject/direct_reduction/{git_sha}/reduction_files/DG_reduction.py",
        "DG_reduction.py",
    )
    get_file_from_request(
        f"https://raw.githubusercontent.com/mantidproject/direct_reduction/{git_sha}/reduction_files/DG_whitevan.py",
        "DG_whitevan.py",
    )
    get_file_from_request(
        f"https://raw.githubusercontent.com/mantidproject/direct_reduction/{git_sha}/reduction_files/DG_monovan.py",
        "DG_monovan.py",
    )
    get_file_from_request(
        "https://raw.githubusercontent.com/pace-neutrons/InstrumentFiles/addf56cf27043dbae5c67a1f26ad76ecf7800324/merlin/mask_24_5.xml",
        "mask_24_5.xml",
    )
    
    
    # END FIA BLOCK
    
    ############ User defined parameters ##############
    ############ User defined parameters ##############
    rotation_block_name = "Rot"
    rotation_bin_size = 5  # In degrees
    
    # For silicon data:
    # runno = 69266
    # wbvan = 69168
    # ei_list = 'auto'
    # lattice_pars = [5.43, 5.43, 5.43]  # In Angstrom
    # lattice_ang = [90, 90, 90]         # In degrees
    # uvector = '1, 0, 0'                # Reciprocal lattice vector parallel to incident beam direction when rotation=psi0
    # vvector = '0, 1, 0'                # Reciprocal lattice vector perpendicular to u in the horizontal plane
    # rotation_zero_angle = -2           # psi0 in Horace notation
    
    
    # For live data on 20250516 (MnO):
    runno = "lives"
    wbvan = 71617
    ei_list = 30  # Auto-ei currently doesn't work for live (streamed) data
    lattice_pars = [4.446, 4.446, 4.446]
    lattice_ang = [90, 90, 90]
    uvector = "1,1,0"
    vvector = "0,0,1"
    rotation_zero_angle = 132.5  # psi0 in Horace notation
    
    ###################################################
    
    output_dir = "/output"  # Do not change
    
    import importlib
    
    sys.path.append(os.path.dirname(__file__))
    
    import reduction_utils
    
    importlib.reload(reduction_utils)
    import mantid
    from mantid.simpleapi import (
        mtd,
        Load,
        MergeMD,
        BinMD,
        CompactMD,
        SaveMD,
        AddTimeSeriesLog,
        SaveNexusProcessed,
    )
    SaveNexusProcessed(Filename="~/work/live-data/output-lives.nxs", InputWorkspace="lives")
    # Default save directory (/output only for autoreduction as the RBNumber/autoreduced dir is mounted here)
    mantid.config["defaultsave.directory"] = output_dir  # data_dir
    
    
    os.environ["EPICS_CA_MAX_ARRAY_BYTES"] = "20000"
    os.environ["EPICS_CA_ADDR_LIST"] = "130.246.39.152:5066"
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    
    wsname = runno
    if wsname not in mtd:
        ws = Load(wsname, OutputWorkspace=wsname)
    else:
        ws = mtd[wsname]
    
    if ei_list == "auto" or (hasattr(ei_list, "__iter__") and ei_list[0] == "auto"):
        ei_list = reduction_utils.autoei(ws)
    if not hasattr(ei_list, "__iter__"):
        ei_list = [ei_list]
    
    # Human-readable NeXus names for the MDHistoWorkspace axes saved below. Mantid names
    # each MD dimension after its Q-component spec ("[H,0,0]", "DeltaE") and SaveMD copies
    # that into the NeXus long_name of data/D<i>, so the viewers show "[H,0,0]" and fall
    # back to D0..D3. Index i must match AlignedDim{i} in the BinMD call below.
    mdhisto_axis_labels = [
        ("H", "H (rtu)", "rtu"),
        ("K", "K (rtu)", "rtu"),
        ("L", "L (rtu)", "rtu"),
        ("E", "Energy transfer (meV)", "meV"),
    ]
    # Extra datasets worth plotting against the same axes (H5Web offers them in its
    # heatmap signal picker).
    mdhisto_aux_signals = ["num_events", "mask"]

    def label_mdhisto_axes(filename):
        """Rewrite the axis metadata of a SaveMD'd MDHistoWorkspace in place.

        Sets long_name/units (used as plot axis titles) and short_name (used on the
        viewer's dimension buttons) on every data/D<i>, names the signal, records a
        default_slice so the viewer opens near E=0 and L=0 rather than on the first bin
        of each sliced dimension, and lists the auxiliary signals.

        Failures are reported but never raised: the reduced data is already on disk, and
        a labelling problem must not abort a live reduction cycle.
        """
        try:
            import h5py
            import numpy as np

            str_array = h5py.special_dtype(vlen=str)

            def bin_centre_nearest_zero(dataset):
                edges = np.asarray(dataset, dtype=float)
                centres = (edges[:-1] + edges[1:]) / 2.0
                return int(np.argmin(np.abs(centres)))

            with h5py.File(filename, "a") as nxs:
                data = nxs["MDHistoWorkspace/data"]
                signal = data["signal"]
                shape = signal.shape

                # Legacy NeXus: the signal's 'axes' attribute lists the boundary datasets
                # in HDF5 (C) order, so axes[i] describes signal dimension i. Read it
                # rather than assume an ordering, and check it before writing anything.
                axes_raw = signal.attrs.get("axes", "")
                if isinstance(axes_raw, bytes):
                    axes_raw = axes_raw.decode()
                axis_names = [n for n in str(axes_raw).replace(",", ":").split(":") if n]

                if len(axis_names) != len(shape):
                    print(
                        f"Axes attribute '{axes_raw}' does not describe signal shape {shape}; "
                        "leaving axis labels untouched",
                        flush=True,
                    )
                    return

                indices = []
                for c_axis, name in enumerate(axis_names):
                    if not name.startswith("D") or not name[1:].isdigit():
                        print(f"Unexpected axis dataset '{name}'; leaving labels untouched", flush=True)
                        return
                    dim_index = int(name[1:])
                    if len(data[name]) - 1 != shape[c_axis] or dim_index >= len(mdhisto_axis_labels):
                        print(
                            f"Axis '{name}' does not match signal shape {shape}; "
                            "leaving axis labels untouched",
                            flush=True,
                        )
                        return
                    indices.append(dim_index)

                for dim_index in indices:
                    short, long_name, unit = mdhisto_axis_labels[dim_index]
                    axis = data[f"D{dim_index}"]
                    axis.attrs["long_name"] = long_name
                    axis.attrs["units"] = unit
                    axis.attrs["short_name"] = short

                # Without this the viewer titles the plot "signal".
                signal.attrs["long_name"] = "S(Q,E) (a.u.)"

                # H5Web slices the leading dimensions and maps the last two to y and x.
                # Each '.' picks the next mapped axis in that same order.
                sliced = [str(bin_centre_nearest_zero(data[n])) for n in axis_names[: len(shape) - 2]]
                data.attrs.create(
                    "default_slice",
                    sliced + ["."] * min(2, len(shape)),
                    dtype=str_array,
                )

                aux = [name for name in mdhisto_aux_signals if name in data]
                if aux:
                    data.attrs.create("auxiliary_signals", aux, dtype=str_array)

            print(f"Labelled MDHistoWorkspace axes of {filename}", flush=True)
        except Exception as error:
            print(f"Could not label axes of {filename}: {error}", flush=True)

    for ei in ei_list:
        output_ws = reduction_utils.iliad(
            runno=wsname,
            wbvan=wbvan,
            ei=ei,
            FixEi=True,
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
            hard_mask_file="mask_24_5.xml",
            inst="MERLIN",
            powdermap="MERLIN_rings_251.xml",
        )
        allws = []
        print(f"{ei:g}meV:")
        print(mtd.getObjectNames())
        for w in mtd.getObjectNames():
            if f"{wsname}_{ei:g}meV" in w and w.endswith("_ang_md"):
                allws.append(w)
        if len(allws) > 1:
            wsout = MergeMD(",".join(allws), OutputWorkspace=f"MER{runno}_{ei:g}meV_1to1_md")
        else:
            wsout = allws[0]
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
        mdhisto_file = os.path.join(output_dir, f"MER{runno}_{ei:g}meV_1to1_mdhisto.nxs")
        SaveMD(wsbin, Filename=mdhisto_file)
        label_mdhisto_axes(mdhisto_file)
        mtd.remove(wsbin.name())
        # mtd.remove(wsout.name())
        for ws in allws:
            mtd.remove(ws)
