from pathlib import Path
from isis_powder.gem import Gem

######
# GEM calibration
######

# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
from os import path

wsname= 'GEM00100655'
Load(Filename=f'{wsname}.nxs', OutputWorkspace=wsname)
ExtractMonitors(InputWorkspace=wsname, DetectorWorkspace=wsname, MonitorWorkspace=f'{wsname}_mon')

# convert the original workspace to d
ws_uncal = ConvertUnits(InputWorkspace=wsname,OutputWorkspace=wsname +'_uncal', Target="dSpacing")

# save the uncalibrated TOF workspace for later
grp = CreateGroupingWorkspace(InputWorkspace=ws_uncal, GroupDetectorsBy='bank', OutputWorkspace='grp')
ws_uncal_foc = DiffractionFocussing(InputWorkspace=ws_uncal, OutputWorkspace=ws_uncal.name() + "_foc", 
                                    GroupingWorkspace='grp', PreserveEvents=False)


######
# Get reference curve and d-spacing
######

dpk = 1.6374  # 311 silicon
dmin, dwidth, dmax = 1.4, 0.002, 1.7
# dpk = 1.9201
# dmin, dwidth, dmax = 1.75, 0.002, 2.05
ws_crop = Rebin(InputWorkspace=ws_uncal, OutputWorkspace=f'{ws_uncal.name()}_crop', Params=f'{dmin},{dwidth},{dmax}')

# fit to get observed d-sapcing
ispec = 6000 # 1800 # 3000
imax = np.argmax(ws_crop.readY(ispec))
xmax = ws_crop.readX(ispec)[imax]
func = IkedaCarpenterPV(I=1000, X0=xmax)  + FlatBackground(A0=0)
func.function.setMatrixWorkspace(ws_crop, ispec, 0.0, 0.0)  # calculate A,B etc.
func.freeAll()
[func.fix(f"f0.{par}") for par in ("Alpha0", "Alpha1", "Beta0", "Kappa", "SigmaSquared", "Gamma")]
res = Fit(Function=func, InputWorkspace=ws_crop, Output=ws_crop.name(), OutputCompositeMembers=True, ConvolveMembers=True, 
            WorkspaceIndex=ispec, Normalise=True)
func = res.Function
[func.free(f"f0.{par}") for par in ("SigmaSquared", "Gamma")]
res = Fit(Function=func, InputWorkspace=ws_crop, Output=ws_crop.name(), OutputCompositeMembers=True, ConvolveMembers=True, 
            WorkspaceIndex=ispec, Normalise=True)
func = res.Function
[func.free(f"f0.{par}") for par in ("Alpha0", "Alpha1", "Beta0")]
res = Fit(Function=func, InputWorkspace=ws_crop, Output=ws_crop.name(), OutputCompositeMembers=True, ConvolveMembers=True, 
            WorkspaceIndex=ispec, Normalise=True)
dobs = res.Function.getFunction(0).getParameterValue('X0')


######
# perform calibration
######
babylon_fpath = r"\\olympic\Babylon5\Public\RWaite"
cross_cor = CrossCorrelate(InputWorkspace=ws_crop, ReferenceSpectra=ispec, XMin=dmin, XMax=dmax,
                           WorkspaceIndexMin=0, WorkspaceIndexMax=ws_crop.getNumberHistograms()-1)
offsets = GetDetectorOffsets(InputWorkspace=cross_cor, Step=dwidth, OffsetMode='Absolute',
                             MaxOffset=1, DReference=dobs, XMin=-200, XMax=200,
                             DIdeal=dpk,PeakFunction='Gaussian')  # GroupingFileName=noffsetfile,
SaveCalFile(Filename=path.join(babylon_fpath, 'offsets_2026_cycle261_RWaite.cal'), OffsetsWorkspace=offsets)

# apply calibration to tof workspace
ApplyDiffCal(InstrumentWorkspace=wsname, OffsetsWorkspace=offsets)
ws_cal = ConvertUnits(InputWorkspace=wsname, OutputWorkspace=wsname + "_cal", Target="dSpacing")
ws_cal_foc = DiffractionFocussing(InputWorkspace=ws_cal, OutputWorkspace=ws_cal.name() + "_foc", 
                                  GroupingWorkspace='grp', PreserveEvents=False)
# apply old calibrationApplyDiffCal(InstrumentWorkspace=wsname, OffsetsWorkspace=offsets)
ApplyDiffCal(InstrumentWorkspace=wsname, ClearCalibration=True)
ApplyDiffCal(InstrumentWorkspace=wsname, CalibrationFile=path.join(babylon_fpath, 'offsets_2023_cycle231.cal'))
ws_cal_old = ConvertUnits(InputWorkspace=wsname, OutputWorkspace=wsname + "_old_cal", Target="dSpacing")
ws_cal_old_foc = DiffractionFocussing(InputWorkspace=ws_cal_old, OutputWorkspace=ws_cal_old.name() + "_foc", 
                                  GroupingWorkspace='grp', PreserveEvents=False)

                
######
# plots
######

dpks = [1.1085, 1.2459, 1.3577, 1.6374, 1.9201, 3.1355] 
fig, axes = plt.subplots(1,2 , sharex=True, sharey=True, subplot_kw={'projection': 'mantid'})
fig.set_figwidth(1.5*fig.get_figwidth())
for iax, ws in enumerate([ws_uncal, ws_cal]):
    cfill = axes[iax].imshow(ws, aspect='auto', cmap='viridis', distribution=False, interpolation='none', label='_child0', origin='lower')
    # cfill.set_norm(plt.Normalize(vmax=0.8*ws_cal.extractY().max()))
    axes[iax].set_title(ws.name())
    for d in dpks:
        axes[iax].axvline(d, color=3*[0.7], ls='--', alpha=0.5)
axes[0].set_xlabel('d-Spacing ($\\AA$)')
axes[0].set_ylabel('Spectrum')
axes[0].set_xlim([1.5 , 3.4])
fig.show()


# plot individual banks
ibank = 2
fig, axes = plt.subplots(subplot_kw={'projection': 'mantid'})
fig.set_figwidth(1.5*fig.get_figwidth())
for ws in [ws_uncal_foc, ws_cal_foc, ws_cal_old_foc]:
    axes.plot(ws,  wkspIndex=ibank)
for d in dpks:
    axes.axvline(d, color =3*[0.7], ls='--')
axes.set_xlim([1.5, 3.35])
legend = axes.legend(fontsize=8.0).set_draggable(True).legend
axes.set_title(f"bank{ibank+1}")
fig.show()


######
# autoreduction
######

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
