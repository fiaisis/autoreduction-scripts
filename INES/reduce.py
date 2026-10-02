# Copyright (C) 2026 ISIS Rutherford Appleton Laboratory UKRI
# SPDX - License - Identifier: GPL-3.0-or-later
from __future__ import annotations
from pathlib import Path

ngem = True
reduce = False
output = ""
output_path = "/output"
runno = 28319
dataset_path = Path("/home/ubuntu/large")
ngem_path = "/ngem/nGEM-INES/DATA/INES_2026_03/INES28319"

if ngem:
    from mantid.simpleapi import LoadNGEM, CropWorkspace, Rebin, MergeRuns, SaveNexusProcessed, DeleteWorkspaces

    # Find each subdir at ngem_path
    subdirs = [f for f in Path(ngem_path).iterdir() if f.is_dir()]

    # Grab all files from subdir
    all_output_ws = []
    for ii, subdir in enumerate(subdirs):
        ws_list = []
        for jj, file in enumerate(subdir.iterdir()):
            # If a .edb load it, crop it, rebin it
            if file.suffix == ".edb":
                print("Loading file: " + str(file))
                ws_name = f"INES{runno}_{jj}"
                LoadNGEM(Filename=str(file), OutputWorkspace=ws_name, GenerateEventsPerFrame=False)
                CropWorkspace(InputWorkspace=ws_name, OutputWorkspace=ws_name, Xmin=0, Xmax=1999)
                Rebin(InputWorkspace=ws_name, OutputWorkspace=ws_name, Params="0,1,1999", PreserveEvents=False)
                ws_list.append(ws_name)
        # Now merge, save, and cleanup
        output_ws = f"INES{runno}"
        MergeRuns(InputWorkspaces=ws_list, OutputWorkspace=output_ws)

        # Grab the output path from ngem_path, for example, /ngem/nGEM-INES/DATA/INES_2026_03/INES28319, turns into
        # /ngem/nGEM-INES/DATA/INES_2026_03, then turned into INES_2026_03 then is split by _ into, INES, 2026, and 03
        _, output_year, output_cycle = Path(ngem_path).parent.stem.split("_")
        output_year = output_year[:-2]
        output_cycle = output_cycle[:-1]
        output_path = Path(ngem_path).parent.parent / f"INES_{output_year}_{output_cycle}_nxs" / f"{output_ws}.nxs"
        print("Outputting merged workspace to: " + str(output_path))

        SaveNexusProcessed(InputWorkspace=output_ws, Filename=str(output_path), PreserveEvents=False)
        DeleteWorkspaces(WorkspaceList=ws_list + list(output_ws))
        all_output_ws.append(output_ws)

    output = all_output_ws

elif reduce:
    from mantid.simpleapi import *
    import numpy as np

    SAMPLE_NAME = "Gold_Go_1"

    OPEN_BEAM_RUNS = [26225]

    SAMPLE_RUNS = [26224]

    # Paths
    PATH_MONITORS = "M:/cycle_25_5/"
    PATH_SAMPLE = "D:/GEM/GEM_2025_05/RUN/"
    SAVE_PATH = "D:/GEM/GEM_2025_05/"


    # =============================================================
    # ---- FUNCTIONS ----------------------------------------------
    # =============================================================

    def load_monitor(run):
        """
        Load a monitor spectrum from a NeXus file, crop and rebin it.
        Returns the processed workspace.
        """
        ws_name = "INS" + str(run)
        Load(Filename=PATH_MONITORS + ws_name + '.nxs', OutputWorkspace=ws_name, SpectrumMin=145, SpectrumMax=145)
        CropWorkspace(InputWorkspace=ws_name, OutputWorkspace=ws_name, Xmin=149, Xmax=1999)
        Rebin(InputWorkspace=ws_name, OutputWorkspace=ws_name, Params="149,10,1999", PreserveEvents=False)
        return ws_name


    def compute_monitor_ratio(openbeam_run, sample_run):
        """
        Compute the ratio of open-beam to sample monitors and return the mean ratio.
        """
        mon_ratio = 'Mon' + str(sample_run) + '_div_' + 'Mon' + str(openbeam_run)
        Divide(LHSWorkspace="INS" + str(openbeam_run), RHSWorkspace="INS" + str(sample_run), OutputWorkspace=mon_ratio,
               WarnOnZeroDivide=1)
        ReplaceSpecialValues(InputWorkspace=mon_ratio, OutputWorkspace=mon_ratio,
                             NaNValue=0, InfinityValue=0,
                             BigNumberThreshold=1e8, BigNumberValue=1e5,
                             BigNumberError=1000)

        dataY = mtd[mon_ratio].extractY()[0]
        mean_ratio = np.mean(dataY)
        std_ratio = np.sqrt(np.mean((dataY - mean_ratio) ** 2))
        print(f"--> Monitor ratio mean: {mean_ratio:.4f} Ãƒâ€šÃ‚Â± {std_ratio:.4f}")
        return mean_ratio, std_ratio, mon_ratio


    def load_data(run, path):
        """
        Load processed raw NeXus data within the TOF region of interest.
        """
        ws = 'INES' + str(run)
        LoadNexusProcessed(Filename=path + ws + '.nxs', OutputWorkspace=ws)
        CropWorkspace(InputWorkspace=ws, OutputWorkspace=ws, Xmin=0, Xmax=1999)
        return ws


    def compute_tranmsission(sample_run, open_run, mean_ratio):
        """
        Normalize one pair of sample data by associated open beam and apply monitors normalisation.
        Returns the normalised workspace.
        """
        transmission = 'INES' + str(sample_run) + '_div_INES' + str(open_run)
        Divide(LHSWorkspace='INES' + str(sample_run), RHSWorkspace='INES' + str(open_run), OutputWorkspace=transmission,
               WarnOnZeroDivide=1)
        ReplaceSpecialValues(InputWorkspace=transmission, OutputWorkspace=transmission,
                             NaNValue=0, InfinityValue=0,
                             BigNumberThreshold=1e8, BigNumberValue=1e5,
                             BigNumberError=1000)

        norm_T = transmission + '_norm_monitors'
        Scale(InputWorkspace=transmission, Factor=mean_ratio, Operation='Multiply', OutputWorkspace=norm_T)
        return norm_T


    def cleanup_workspaces(workspace_names):
        """Delete temporary Mantid workspaces."""
        for ws in workspace_names:
            if mtd.doesExist(ws):
                DeleteWorkspace(Workspace=ws)

        # =============================================================
        # ---- EXECUTION ----------------------------------------------
        # =============================================================

        """
        Process one sample/open-beam pair:
        1. Load monitors
        2. Compute monitor ratio mean
        3. Load data, compute transmission, normalize
        4. Return normalized workspace name
        """


    merge_list = []
    for sample_run, openbeam_run in zip(SAMPLE_RUNS, OPEN_BEAM_RUNS):
        print(f"\n--- Processing pair: Sample {sample_run} / Open Beam {openbeam_run} ---")

        # Load and preprocess monitors
        print('Load Open Beam monitor: INS', openbeam_run)
        load_monitor(openbeam_run)
        print('Load Sample Monitor: INS', sample_run)
        load_monitor(sample_run)

        # Compute Monitors ratio
        mean_ratio, std_ratio, mon_ratio = compute_monitor_ratio(openbeam_run, sample_run)

        # Load NRTI data
        print('Load Sample-in: INES', sample_run)
        load_data(sample_run, PATH_SAMPLE)
        print('Load Open Beam: INES', openbeam_run)
        load_data(openbeam_run, PATH_SAMPLE)

        # Normalise
        norm_ws = compute_tranmsission(sample_run, openbeam_run, mean_ratio)

        # Clean up temporary workspaces
        cleanup_workspaces([f"INS{sample_run}", f"INS{openbeam_run}",
                            f"INES{sample_run}", f"INES{openbeam_run}",
                            f"INES{sample_run}_div_INES{openbeam_run}",
                            mon_ratio])

        merge_list.append(norm_ws)

    MergeRuns(InputWorkspaces=merge_list, OutputWorkspace=str(SAMPLE_NAME) + '/Empty_normMonitor')
    NoFile = 1 / len(merge_list)
    Scale(InputWorkspace=str(SAMPLE_NAME) + '/Empty_normMonitor', Factor=NoFile, Operation='Multiply',
          OutputWorkspace=str(SAMPLE_NAME) + '/Empty_normMonitor')
    for file in merge_list:
        DeleteWorkspace(Workspace=file)

    SaveNexusProcessed(InputWorkspace=str(SAMPLE_NAME) + '/Empty_normMonitor',
                       Filename=SAVE_PATH + str(SAMPLE_NAME) + '_norm.nxs')

    print("\nNormalization complete!")