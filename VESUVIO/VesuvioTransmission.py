from mantid.kernel import *
from mantid.api import *
from mantid.simpleapi import *
import numpy as np



class VesuvioTransmission(PythonAlgorithm):

    def summary(self):
        return ("This algorithm evaluates the transmission spectrum on the VESUVIO spectrometer for measured sample and empty run numbers.")

    def category(self):
        return 'MyTools'

    def PyInit(self):
        self.declareProperty('OutputWorkspace', "", StringMandatoryValidator(), direction=Direction.Input, doc="The root-name for the output workspaces")
        self.declareProperty("Runs", "", StringMandatoryValidator(), direction=Direction.Input, doc="The root-name for the output workspaces")
        self.declareProperty("EmptyRuns", "", StringMandatoryValidator(), direction=Direction.Input, doc="The root-name for the output workspaces")
        self.declareProperty("Grouping","SumOfAllRuns",StringListValidator(["TimeScan","SumOfAllRuns"]))
        self.declareProperty("Target","Energy",StringListValidator(["Energy","Wavelength"]))
        self.declareProperty("Rebin",False)
        self.declareProperty('RebinParameters', "0.6,-0.005,1.e7", StringMandatoryValidator(), direction=Direction.Input, doc="Rebin parameters")
        self.declareProperty("CalculateXS",False)
        self.declareProperty("InvertMonitors",False)
        self.declareProperty("SmoothIncidentSpectrum",False)
    def PyExec(self):
        invert=self.getProperty("InvertMonitors").value
        smooth=self.getProperty("SmoothIncidentSpectrum").value
        grouping=self.getProperty("Grouping").value
        target=self.getProperty("Target").value
        reb_bool=self.getProperty("Rebin").value
        reb_parameters=self.getProperty("RebinParameters").value
        runs = self.getProperty("Runs").value
        if "-" in runs:
            lower, upper = runs.split("-")
        empty_runs = self.getProperty("EmptyRuns").value
        name = self.getProperty("OutputWorkspace").value
        name=name+'_transmission'
        calculate_cross_section=self.getProperty("CalculateXS").value
        if grouping=="SumOfAllRuns":
            LoadVesuvio(Filename=runs, OutputWorkspace='sample', SpectrumList='3-134',Mode='FoilOut',SumSpectra=True,LoadMonitors=True)
            LoadVesuvio(Filename=empty_runs, OutputWorkspace='empty', SpectrumList='3-134',Mode='FoilOut',SumSpectra=True,LoadMonitors=True)
            RebinToWorkspace(WorkspaceToRebin="empty_monitors",WorkspaceToMatch="sample_monitors", PreserveEvents=True, OutputWorkspace="empty_monitors")
            Divide(LHSWorkspace='sample_monitors', RHSWorkspace='empty_monitors', OutputWorkspace=name)
            ConvertUnits(InputWorkspace=name, OutputWorkspace=name, Target=target, EMode='Elastic')
            if reb_bool:
                if (target=="Energy"):
                    Rebin(InputWorkspace=name,Params=reb_parameters,OutputWorkspace=name,FullBinsOnly=True,PreserveEvents=True)
            ExtractSingleSpectrum(InputWorkspace=name, OutputWorkspace='tmp', WorkspaceIndex=0)
            ExtractSingleSpectrum(InputWorkspace=name, OutputWorkspace=name, WorkspaceIndex=1)
            RebinToWorkspace(WorkspaceToRebin='tmp',WorkspaceToMatch=name,OutputWorkspace='tmp')
            if not invert :
                if smooth :
                    SmoothData(InputWorkspace='tmp',NPoints=5,OutputWorkspace='tmp')
                Divide(LHSWorkspace=name, RHSWorkspace='tmp', OutputWorkspace=name)
            else :
                if smooth :
                    SmoothData(InputWorkspace=name,NPoints=5,OutputWorkspace=name)
                Divide(LHSWorkspace='tmp', RHSWorkspace=name, OutputWorkspace=name)
        elif grouping=="TimeScan":
            upper=int(upper)
            lower=int(lower)
            LoadVesuvio(Filename=str(lower), OutputWorkspace='sample', SpectrumList='3-134',Mode='FoilOut',SumSpectra=True,LoadMonitors=True)
            LoadVesuvio(Filename=empty_runs, OutputWorkspace='empty', SpectrumList='3-134',Mode='FoilOut',SumSpectra=True,LoadMonitors=True)          
            Divide(LHSWorkspace='sample_monitors', RHSWorkspace='empty_monitors', OutputWorkspace=name)
            ConvertUnits(InputWorkspace=name, OutputWorkspace=name, Target='Energy', EMode='Elastic')
            ExtractSingleSpectrum(InputWorkspace=name, OutputWorkspace='tmp', WorkspaceIndex=0)
            ExtractSingleSpectrum(InputWorkspace=name, OutputWorkspace=name, WorkspaceIndex=1)
            RebinToWorkspace(WorkspaceToRebin='tmp',WorkspaceToMatch=name,OutputWorkspace='tmp')
            Divide(LHSWorkspace=name, RHSWorkspace='tmp', OutputWorkspace=name)
            for runs in range(lower+1, upper+1):
                print(runs)
                LoadVesuvio(Filename=str(runs), OutputWorkspace='sample', SpectrumList='3-134',Mode='FoilOut',SumSpectra=True,LoadMonitors=True)
                Divide(LHSWorkspace='sample_monitors', RHSWorkspace='empty_monitors', OutputWorkspace='tmp')
                ConvertUnits(InputWorkspace='tmp', OutputWorkspace='tmp', Target='Energy', EMode='Elastic')
                ExtractSingleSpectrum(InputWorkspace='tmp', OutputWorkspace='tmp2', WorkspaceIndex=0)
                ExtractSingleSpectrum(InputWorkspace='tmp', OutputWorkspace='tmp', WorkspaceIndex=1)
                RebinToWorkspace(WorkspaceToRebin='tmp2',WorkspaceToMatch='tmp',OutputWorkspace='tmp2')
                Divide(LHSWorkspace='tmp', RHSWorkspace='tmp2', OutputWorkspace='tmp')
                AppendSpectra(InputWorkspace1=name, InputWorkspace2='tmp',OutputWorkspace=name)
            DeleteWorkspace('tmp2')
        if calculate_cross_section:
            Logarithm(InputWorkspace=name,OutputWorkspace=name+'_XS')
            Scale(InputWorkspace=name+'_XS',Factor=-1,Operation="Multiply",OutputWorkspace=name+'_XS')
            Integration(InputWorkspace=name+'_XS',RangeLower=1000,RangeUpper=11000,OutputWorkspace='tmp')
            Divide(LHSWorkspace=name+'_XS',RHSWorkspace='tmp',OutputWorkspace=name+'_XS')
            Scale(InputWorkspace=name+'_XS',Factor=10000,OutputWorkspace=name+'_XS')
        DeleteWorkspace('empty')
        DeleteWorkspace('empty_monitors')
        DeleteWorkspace('sample')
        DeleteWorkspace('sample_monitors')
        DeleteWorkspace('tmp')


AlgorithmFactory.subscribe(VesuvioTransmission)
