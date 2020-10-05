#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements the EM algorithm and Variants for error rates estimation from syndromes in concatenated quantum codes.
Also provides methods for simulations of this algorithm on our cluster, using some parallelization
"""
import CustomBP as bp
import QueueParallelization

import numpy as np

import math
import json
import pickle
import copy
import datetime
import sys
import os

import multiprocessing as mp

"""
Standard Expectation Maximization Algorithm for Error Rates estimation (for EM compare [Koller, Probabilistic Graphical Models:Principles and Techniques])
"""
def NormalizeRows(a):
    """
    Helper
    """
    if a is not None:
        row_sums = a.sum(axis = 1)
        return a  / row_sums[:,np.newaxis]
    else:
        return None
def EM(Code,SyndromeData,Iterations,InitQubitRates = None,QubitPriorStatistics = None,InitMeasurementRates = None, MeasurementPriorStatistics = None):
    """
    Perform EM on given ConcatQCode with given Syndrome Data
    Both returns the the Estimated Rates of all iterations, and updates the rates in the ConcatQCode object that is passed, such that after calling this method it has the final rates
    
    Can specify a Dirichlet Prior to perform MAP Estimation, by passing to PriorStatistics a vector of "Pseudocounts" for each parameter.
    PriorStatistics = np.zeros((Code.n_physical_Qubits,4)) corresponds to maximum likelihood estimation for a code without measurement errors
    e.g. PriorStatistics = [[2,1,1,1]] would correspond to a dirichlet prior for a single qubit that is biased towards the identity error.
    """
    if QubitPriorStatistics is None:
        QubitPriorStatistics = np.zeros((Code.n_physical_Qubits,4))
    if Code.HasMeasurementErrors:
        if MeasurementPriorStatistics is None:
            MeasurementPriorStatistics = np.zeros((Code.n_total_checks,2))
    QubitRates = [] #The Predicted Qubit Error Rates over the Iterations
    MeasurementRates = [] # The Predicted Measurement Error Rates over the Iterations
    Code.SetErrorRates(InitQubitRates,InitMeasurementRates) #If one of them is None then we just keep the corresponding rates that are currently set on the code.
    qrate,measrate = Code.GetRates(flat=True)
    QubitRates += [qrate]
    MeasurementRates += [measrate]
    SyndromeCounts = GetSyndromeCounts(SyndromeData)
    for iteration in range(Iterations):
        #Expectation Step
        SyndromeMarginals = CalculateMarginals(SyndromeCounts, Code)
        QubitSufficientStatistics = QubitPriorStatistics
        MeasurementSufficientStatistics = MeasurementPriorStatistics
        for s in SyndromeCounts.keys():
            QubitSufficientStatistics = QubitSufficientStatistics + SyndromeMarginals[s][0]*SyndromeCounts[s]
            if Code.HasMeasurementErrors:
                MeasurementSufficientStatistics = MeasurementSufficientStatistics + SyndromeMarginals[s][1]*SyndromeCounts[s]
        #Normalize
        QubitSufficientStatistics = NormalizeRows(QubitSufficientStatistics)
        if Code.HasMeasurementErrors:
            MeasurementSufficientStatistics = NormalizeRows(MeasurementSufficientStatistics)
        if np.any(np.isnan(QubitSufficientStatistics)):
            """
            Chose convention that we keep old error rates if nan is encountered
            This can happen for very low / 0 error rates
            """
            QubitSufficientStatistics = Code.ErrorRates
            print("Encountered Nan")
        if Code.HasMeasurementErrors:
            if np.any(np.isnan(MeasurementSufficientStatistics)):
                """
                Chose convention that we keep old error rates if nan is encountered
                This can happen for very low / 0 error rates
                """
                MeasurementSufficientStatistics = Code.GetRates(flat=True)[1] #The old measurement error rates in appropriate shape
                print("Encountered Nan")
        #Maximization step
        Code.SetErrorRates(QubitSufficientStatistics,MeasurementSufficientStatistics)
        qrate,measrate = Code.GetRates(flat=True)
        QubitRates += [qrate]
        MeasurementRates += [measrate]
    return QubitRates,MeasurementRates
def GetSyndromeCounts(SyndromeData):
    """
    Counts the number of occurences of each syndrome in the data set
    """
    SyndromeCounts = {}
    for syndrome in SyndromeData:
        s = tuple(syndrome)
        if s in SyndromeCounts:
            SyndromeCounts[s] += 1
        else:
            SyndromeCounts[s] = 1
    return SyndromeCounts      
def CalculateMarginals(SyndromeCounts,Code):
    """
    Calculates the marginals of all qubits and measurement error nodes for each syndrome in the data set
    Stores two sets of marginals for each syndrome: One correspondign to qubit errors, one corresponding to measurement errors
    """
    SyndromeMarginals = {}
    #Calculate all Marginals
    for syndrome in SyndromeCounts.keys():
        s = np.array(syndrome)
        Code.RunBP(s)
        Marginals = Code.LeafMarginals()
        QubitMarginals = np.array(Marginals[0:Code.n_physical_Qubits])
        MeasurementMarginals = []
        if Code.HasMeasurementErrors:
            MeasurementMarginals =np.array(Marginals[Code.n_physical_Qubits:])
        SyndromeMarginals[syndrome] = (QubitMarginals,MeasurementMarginals)
    return SyndromeMarginals


#%%
"""
Hard Assignment EM (compare [Koller, Probabilistic Graphical Models:Principles and Techniques])
"""
def HardAssignmentEM(Code,SyndromeData,Iterations,InitQubitRates = None,QubitPriorStatistics = None,InitMeasurementRates = None, MeasurementPriorStatistics = None):
    """
    Note that this must have the same signature as EM for compatibility with test methods
    
    Can specify a Dirichlet Prior to perform MAP Estimation, by passing to PriorStatistics a vector of "Pseudocounts" for each parameter.
    PriorStatistics = np.zeros((Code.n_physical_Qubits,4)) corresponds to maximum likelihood estimation
    e.g. PriorStatistics = [[2,1,1,1]] would correspond to a Dirichlet prior for a single qubit that is biased towards the identity error.
    """
    n_qubits = Code.n_physical_Qubits
    n_check = Code.n_total_checks
    if QubitPriorStatistics is None:
        QubitPriorStatistics = np.zeros((Code.n_physical_Qubits,4))
    if Code.HasMeasurementErrors:
        if MeasurementPriorStatistics is None:
            MeasurementPriorStatistics = np.zeros((Code.n_total_checks,2))
    Code.SetErrorRates(InitQubitRates,InitMeasurementRates) #If one of them is None then we just keep the corresponding rates that are currently set on the code.
    QubitRates = [] #The Predicted Qubit Error Rates over the Iterations
    MeasurementRates = [] # The Predicted Measurement Error Rates over the Iterations
    qrate,measrate = Code.GetRates(flat=True)
    QubitRates += [qrate]
    MeasurementRates += [measrate]
    SyndromeCounts = GetSyndromeCounts(SyndromeData)
    n_data = SyndromeData.shape[0]   
    for iteration in range(Iterations):
        SyndromeMAPS = CalculateMAPS(SyndromeCounts, Code)
        QubitSufficientStatistics = QubitPriorStatistics
        MeasurementSufficientStatistics = MeasurementPriorStatistics
        for s in SyndromeCounts.keys():
            e = SyndromeMAPS[s]
            for i in range(e.size):
                if i<n_qubits:
                    QubitSufficientStatistics[i,e[i]]+=SyndromeCounts[s]
                else: #This shoudl only happen if the code has measurement errors
                    MeasurementSufficientStatistics[i-n_qubits,e[i]]+=SyndromeCounts[s]
        #Normalize
        QubitSufficientStatistics /= n_data
        if Code.HasMeasurementErrors:
            MeasurementSufficientStatistics /= n_data
        if np.any(np.isnan(QubitSufficientStatistics)):
            """
            Chose convention that we keep old error rates if nan is encountered
            """
            QubitSufficientStatistics = Code.ErrorRates
            print("Encountered Nan")
        if Code.HasMeasurementErrors:
            if np.any(np.isnan(MeasurementSufficientStatistics)):
                """
                Chose convention that we keep old error rates if nan is encountered
                """
                MeasurementSufficientStatistics = Code.GetRates(flat=True)[1] #The old measurement error rates in appropriate shape
                print("Encountered Nan")
        #Maximization step
        Code.SetErrorRates(QubitSufficientStatistics,MeasurementSufficientStatistics)
        qrate,measrate = Code.GetRates(flat=True)
        QubitRates += [qrate]
        MeasurementRates += [measrate]
    return QubitRates,MeasurementRates
def CalculateMAPS(SyndromeCounts,Code):
    """
    Calculate the most likely error for each syndrome
    """
    MAPConfigurations = {}
    #Calculate all Marginals
    for syndrome in SyndromeCounts.keys():
        s = np.array(syndrome)
        Code.RunMaxSum(s)
        MAPError = Code.MAPConfig()
        MAPConfigurations[syndrome] = MAPError
    return MAPConfigurations


#%% Testing of EM via Decoding
class TestDecoderEMPrior:
    """
    Different possibilities of choosing a Dirichlet prior for MAP estimation via EM 
    Different Prior Types are possible:
    "None": Perform Simple Maximum Likelihood Estimation
    "AroundInit": will choose a Prior around the initial Guess Error Rates using a total number of Pseudocounts n_prior_counts (per qubit / measurement bit)
    "Uniform": will choose a uniform Prior using a total number of Pseudocounts n_prior_counts (per qubit / measurement bit)
    """
    def __init__(self,Type,n_prior_counts):
        self.Type = Type
        self.n_prior_counts=n_prior_counts
    def ToDict(self):
        Params = {}
        Params["Type"] = self.Type
        Params["n_prior_counts"] = self.n_prior_counts
        return Params
    def GetPrior(self,InitQubitErrorRates,n_qubits,InitMeasurementErrorRates = None,n_meas=None):
        """
        GuessRates -> Error Rates for qubits, should be in correct shape (n_qubits,4)
        MeasurementGuessRates -> Error Rates for Measurement Errors, should be passed in flattened shape
        QubitPrior is returned in proper shape if the rates are passed in proper shape, but MeasurementPrior is flattened because reshaping into a ragged array is difficult
        """
        if InitMeasurementErrorRates is not None:
            n_meas = len(InitMeasurementErrorRates)
        QubitPriorValues = None
        MeasurementPriorValues = None
        if self.Type == "None":
            QubitPriorValues = None
            MeasurementPriorValues = None
        elif self.Type == "AroundInit":
            QubitPriorValues = InitQubitErrorRates*self.n_prior_counts
            if MeasurementPriorValues is not None:
                MeasurementPriorValues = InitMeasurementErrorRates * self.n_prior_counts #np.concatenate flattens the list into a numpy array
        elif self.Type == "Uniform":
            QubitPriorValues = np.ones((n_qubits,4))*self.n_prior_counts / 4 #Divide by 4 so total prior count is n_prior_counts, no tobservations of each event
            if InitMeasurementErrorRates is not None:
                MeasurementPriorValues = np.ones((n_meas,2))*self.n_prior_counts / 2
        else:
            raise ValueError("Not a valid prior type")
        return QubitPriorValues,MeasurementPriorValues
class TestDecoderParameters:
    """
    A struct that holds all the parameters used in Decoder + EM simulations
    """
    def __init__(self,n_tries = 100, n_Iterations = 1,DecodeAtIterations = None,n_concatenations = 1,p_mean = 0.1/3, Pseudocounts = 20,n_Estimation_data = 10**3,n_Test_Data = 10**4,Printperiod = math.inf, Faithful = False,FixRealRates = True,UseHardAssignments = False, PriorType = "AroundInit", n_prior_counts = 20,p_mean_measurement=0.1):
        """
        Using p_mean_measurement = None corresponds to no measurement error nodes. p_mean_measurement = 0 still creates them, which slows down computation.
        """
        self.n_tries = n_tries
        self.n_Iterations = n_Iterations
        if DecodeAtIterations is None:
            self.DecodeAtIterations = [0,n_Iterations]
        else:
            self.DecodeAtIterations = DecodeAtIterations #An array specifying after which iterations of EM to estimate the logical error rates. [0,n_Iterations] to only estimate logical error rate before any EM steps and final logical error rates
        self.n_concatenations = n_concatenations        
        self.p_mean = p_mean
        self.p_mean_measurement=p_mean_measurement
        self.Pseudocounts = Pseudocounts #Pseudocounts for the Dirichlet drawing the random error rates of the real model. This is NOT a prior used for map estimation.
        self.n_Test_Data = n_Test_Data
        self.n_Estimation_data = n_Estimation_data
        self.Faithful = Faithful #If faithful is true uses the actual rates for decoding, used to estimate the error rate of the decoder with perfect knowledge of the physical error rates
        self.FixRealRates = FixRealRates #Whether to fix the actual error rates and randomly chose the initialization [True] or the other way around [False]
        self.UseHardAssignments = UseHardAssignments # Whether to use normal of hard assignment EM
        self.Printperiod = Printperiod
        #Info about prior for EM
        self.EMPrior = TestDecoderEMPrior(Type = PriorType,n_prior_counts=n_prior_counts)#Prior for doing Map Estimation 
    def ToDict(self):
        Params = {}
        Params["n_tries"] = self.n_tries
        Params["n_Iterations"] = self.n_Iterations
        Params["DecodeAtIterations"] = self.DecodeAtIterations
        Params["n_concatenations"] = self.n_concatenations
        Params["p_mean"] = self.p_mean
        Params["p_mean_measurement"] = self.p_mean_measurement
        Params["Pseudocounts"] = self.Pseudocounts
        Params["n_Test_Data"] = self.n_Test_Data
        Params["n_Estimation_Data"] = self.n_Estimation_data
        Params["Printperiod"] = self.Printperiod
        Params["Faithful"] = self.Faithful
        Params["FixRealRates"] = self.FixRealRates
        Params["UseHardAssignments"] = self.UseHardAssignments
        Params["EMPriorInformation"] = self.EMPrior.ToDict()
        return Params
    def ToJson(self):
        Params = self.ToDict()
        return json.dumps(Params)
    def Copy(self):
        return copy.copy(self)

def Decode(Model,Syndromes,Printperiod = math.inf):
    """
    Implementation of the Maximum Likelihood Decoder
    """
    LogErrs = np.zeros(Syndromes.shape[0],dtype = int)
    for i,s in enumerate(Syndromes):
        Model.BPUpwardsPass(s)
        LogErr = np.argmax(Model.TopMarginal())
        LogErrs[i] = LogErr
        if (i+1) % Printperiod == 0:
            print(str(i+1) + " Errors Decoded")      
            sys.stdout.flush()
    return LogErrs
def TestDecoderParallel(n_test,GuessModel,RealModel,Printperiod = math.inf):
    """
    Tests the Decoder (that assumes rates from GuessModel) on a random errors generated from RealModel and computes the logical error rate
    Parallelized by dividing the test set into chunks and decoding them in parallel using _TestDecoderChunk
    """
    if n_test > 0:
        print("Decoding")
        n_concat = GuessModel.n_concatenations
        GuessRates,MeasurementGuessRates = GuessModel.GetRates(flat = True)
        RealRates,MeasurementRealRates = RealModel.GetRates(flat=True)
        n_processes = mp.cpu_count()
        print("CPU count: ", n_processes)
        chunk_sizes = _DistributeWork(n_test, n_processes)
        print("Chunksizes: ", chunk_sizes)
        with mp.Pool(n_processes) as pool:
            proc = [pool.apply_async(_TestDecoderChunk,args=(n, GuessRates, RealRates,MeasurementGuessRates,MeasurementRealRates, n_concat,Printperiod)) for n in chunk_sizes]
            res = [p.get() for  p in proc]
        TotalSuccessrate = 0
        for n,s in zip(chunk_sizes,res):
             TotalSuccessrate += n*s
        TotalSuccessrate /= n_test
        return TotalSuccessrate
    else:
        return 0
def _TestDecoderChunk(n_test,GuessRates,RealRates,MeasurementGuessRates,MeasurementRealRates,n_concatenations,Printperiod = math.inf):
    """
    Helper for TestDecoder. Decodes one chunk of the data. Multiple instances of this are called in parallel.
    """
    if n_test > 0:
        BaseCode = bp.Create5QubitPerfectCode()
        GuessModel = bp.ConcatQCode(BaseCode,n_concatenations,GuessRates,MeasurementGuessRates)
        RealModel = bp.ConcatQCode(BaseCode, n_concatenations,RealRates,MeasurementRealRates)
        Syndromes,LogErrs,PhysErrs = RealModel.GenerateData(n_test)
        DecodedErrs = Decode(GuessModel,Syndromes,Printperiod)
        SucessRate = np.count_nonzero(np.equal(DecodedErrs,LogErrs)) / n_test
        return  SucessRate
    else:
        return 0
#Distribute the number of simulations as evenly as possible between processors
def _DistributeWork(n_work,n_processes):
    chunksize = math.floor(n_work / n_processes)
    remainder = n_work - chunksize*n_processes
    Chunksizes = [chunksize]*n_processes
    for i in range(remainder):
        Chunksizes[i] += 1
    return Chunksizes

def EMWithDecoding(GuessModel,RealModel,Syndromes,Parameters):
    """
    Performs EM on the given syndrome set to estimate the error rates of the code.
    Estimates the logcial error rate of the decoder at the given iterations of EM. (Using independently generated Syndromes)
    Note that Parameters.DecodeAtIterations should be in ascending order and include 0 as the first entry if you want to estimate the initial error rates
    Returns the estimated error rates of each iteration and the estimated logical error rates of the given iterations.
    """
    #Get the prior 
    qrates,mrates = GuessModel.GetRates(flat=True)
    PriorRates,MeasurementPriorRates = Parameters.EMPrior.GetPrior(qrates,GuessModel.n_physical_Qubits,mrates,GuessModel.n_total_checks)
    #Convert to format that tells you how many additional iterations to do in each step before decoding instead of the absolute number of the iteration
    DecodeAtIterations = Parameters.DecodeAtIterations
    Iterations = Parameters.n_Iterations
    DecodeForIterations = [DecodeAtIterations[0]] + [DecodeAtIterations[i] - DecodeAtIterations[i-1] for  i in range(1,len(DecodeAtIterations))]
    if Parameters.UseHardAssignments == False:
        EMFunction = EM
    else:
        EMFunction = HardAssignmentEM
    prates,mrates = GuessModel.GetRates(flat=True)
    EstimatedPhysicalRates = [prates]
    EstimatedMeasurementRates = [mrates]
    EstimatedSuccessRates = []
   # print(DecodeForIterations)
    for it in DecodeForIterations:
        EstRates,MeasEstRates = EMFunction(GuessModel,Syndromes,it,QubitPriorStatistics=PriorRates,MeasurementPriorStatistics=MeasurementPriorRates)
        #print(EstRates)
        EstimatedPhysicalRates += EstRates[1:] #Exclude first entry since its identical to last entry of previous estimation round
        EstimatedMeasurementRates += MeasEstRates[1:]
        SuccessRate = TestDecoderParallel(Parameters.n_Test_Data,GuessModel,RealModel,Printperiod = Parameters.Printperiod)
        EstimatedSuccessRates += [SuccessRate]

    it = Iterations - DecodeAtIterations[-1]
    EstRates,MeasEstRates = EMFunction(GuessModel,Syndromes,it,QubitPriorStatistics=PriorRates,MeasurementPriorStatistics=MeasurementPriorRates)
    EstimatedPhysicalRates += EstRates[1:]
    EstimatedMeasurementRates += MeasEstRates[1:]
    return np.array(EstimatedPhysicalRates),np.array(EstimatedMeasurementRates),np.array(EstimatedSuccessRates)
#%%
"""
Simulations of Decoder + EM
"""
#Note: My p_mean must be 1/3 of the p given in  [Poulin] because of different conventions for the depolarizing channel
def TestDecoderEstimation(Parameters):
    """
    Sets up actual and guess error rates and performs EM + Decoding. Returns the Estimated Rates (of each Iteration) and the estimated Logical Error Rates (only for Iterations in Parameters.DecodeAtIterations)
    Parameters should be an Object of TestDecoderParameters
    """
    RealModel,GuessModel = Setup5QubitModels(n_concat=Parameters.n_concatenations,p_mean=Parameters.p_mean,p_mean_measurement=Parameters.p_mean_measurement,Pseudocounts=Parameters.Pseudocounts,FixRealRates = Parameters.FixRealRates)
    #Parameters.EMPrior.SetPriorValues(GuessModel.ErrorRates) #Set the prior Values
    if Parameters.Faithful == False:
        Syndromes,Errs,PhysErrs = RealModel.GenerateData(Parameters.n_Estimation_data)    
    if Parameters.Faithful == False:
        #InitSucessRate = TestDecoderParallel(Parameters.n_Test_Data,GuessModel,RealModel,Printperiod = Parameters.Printperiod)
        EstimatedRates,EstimatedMeasRates,SuccessRates = EMWithDecoding(GuessModel,RealModel,Syndromes,Parameters)
    else:
        SuccessRates = np.array([TestDecoderParallel(Parameters.n_Test_Data,RealModel,RealModel)])
        EstimatedRates = []
        EstimatedMeasRates = []
    if Parameters.n_Test_Data > 0:
        print("Initial Error Rate: ", bp.FormatScientific(1- SuccessRates[0]))
        if Parameters.Faithful == False:
            print("Error Rate after Estimation: ", bp.FormatScientific(1 - SuccessRates[-1]))
    # RealPhysicalRates = RealModel.ErrorRates
    # RealMeasRates = RealModel.MeasurementErrorRates
    RealPhysicalRates,RealMeasRates = RealModel.GetRates(flat=True)
    FailureRates = 1 - SuccessRates
    return FailureRates,RealPhysicalRates,RealMeasRates,EstimatedRates,EstimatedMeasRates
def _TestDecoderEstimationChunk(n,Parameters):
    """
    Deprecated, for parallelization via chunks
    """
    Results = []
    for i in range(n):
        print("Starting Estimation: ", i)
        sys.stdout.flush()
        res = TestDecoderEstimation(Parameters)
        Results += [res]
    return Results
class ErrorRatesEstimationTask():
    """
    For Parallelization via Queues
    Parameters: An instance of TestDecoderParameters
    """
    def __init__(self,Parameters):
        self.Parameters = Parameters
    def __call__(self):
        print("Starting Estimation")
        sys.stdout.flush()
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little')) #Reseed the Random number generator in each new process since otherwise you get same sequences
        res = TestDecoderEstimation(self.Parameters)
        #print("Estimation Finished")
        sys.stdout.flush()
        return res
def TestDecoderEstimationMultiple(Parameters,SaveFolder,name):
    """
    This is the best method to actually call for simulations
    Runs multiple simulations of Decoding + EM (using different initialization / actual rates in each one) and returns the data for each. (This is how the box plots in paper are created)
    Depending on the Value of Parameters.n_Test_Data either the simulations are run sequentially but the decoding is parallelized or the simulations are run in parallel.
    """
    #Containers for Logical Error Rates, and Real and Estimated rates of Qubit and Measurement errors
    LogicalErrorRates = []
    RealPhysicalRates = []
    RealMeasurementRates = []
    EstimatedPhysicalRates = []
    EstimatedMeasurementRates = []
    print("SaveFolder: ", SaveFolder, "Name : ", name)
    if os.path.isfile(SaveFolder+"/"+name+".pkl"):
        raise ValueError("Existing Data would be overwritten by this Simulation")    
    try:
        f = open(SaveFolder+"/"+name+".pkl","wb")
        f.close()
    except:
        raise ValueError("SaveFolder could not be opened")
    print("Parameters: ")
    print(Parameters.ToDict())
    
    if Parameters.n_Test_Data > 0:
        """
        Parallelization is over the Test Data Set in this case each time the decoder is tested, no parallelization during EM
        """
        print("Simulations with Decoding")
        for i in range(Parameters.n_tries):
            print()
            print("Estimation Nr: ", i)
            LogicalErr,RealPhysRates,RealMeasRates,EstRates,EstMeasRates = TestDecoderEstimation(Parameters)
            LogicalErrorRates += [LogicalErr]
            RealPhysicalRates += [RealPhysRates]
            RealMeasurementRates += [RealMeasRates]
            EstimatedPhysicalRates += [EstRates]
            EstimatedMeasurementRates+=[EstMeasRates]
    else:
        """
        Parallelize over the Simulations
        """
        print("Simulations without Decoding")
        n_processes = mp.cpu_count()
        print("n_cpu: ", n_processes)
        TaskQueue  =  mp.JoinableQueue()
        ResultQueue = mp.Queue()
        SimulationWorkers = [QueueParallelization.QueueWorker(TaskQueue,ResultQueue,Print=True) for i in range(n_processes)]
        for i in range(Parameters.n_tries):
            TaskQueue.put(ErrorRatesEstimationTask(Parameters))
        for i in range(n_processes): #Put termination signals for the processes
            TaskQueue.put(None)
        for worker in SimulationWorkers:
            worker.start()
        TaskQueue.join()
        Results = [ResultQueue.get() for i in range(Parameters.n_tries)] #It is important that this is done before joining the processes, see:https://stackoverflow.com/questions/26025486/python-processes-not-joining
        print("TaskQueue joined")
        #Close all workers
        for worker in SimulationWorkers:
            worker.join()
        print("Workers closed")
        # Results = list(itertools.chain.from_iterable(res)) #Concatenate the results
        LogicalErrorRates = [r[0] for r in Results]
        RealPhysicalRates = [r[1] for r in Results]
        RealMeasurementRates = [r[2] for r in Results]
        EstimatedPhysicalRates = [r[3] for r in Results]
        EstimatedMeasurementRates = [r[4] for r in Results]
        
    LogicalErrorRates = np.array(LogicalErrorRates)
    RealPhysicalRates = np.array(RealPhysicalRates)
    EstimatedPhysicalRates = np.array(EstimatedPhysicalRates)
    RealMeasurementRates = np.array(RealMeasurementRates)
    EstimatedMeasurementRates = np.array(EstimatedMeasurementRates)
    
    Gains = LogicalErrorRates[:,-1] - LogicalErrorRates[:,0]
    AverageGain = np.sum(Gains) / Gains.size
    StdDevGain = np.std(Gains,dtype = np.float64  ,ddof = 1)
    #RelativeImprovements = 1 - FinalRates / InitialRates
    RelativeImprovements = (LogicalErrorRates[:,0] - LogicalErrorRates[:,-1]) / LogicalErrorRates[:,0]
    AverageRelativeImprovement = np.sum(RelativeImprovements) / RelativeImprovements.size
    print("Improvement: ", AverageGain , " +- ", StdDevGain)
    print("Average Relative Improvement: ", AverageRelativeImprovement)
    InitialLogicalErrorRates = LogicalErrorRates[:,0]
    FinalLogicalErrorRates = LogicalErrorRates[:,-1] #Initial and FinalLogicalErrorRates are just for backwards compatibility with old plotting scripts, in principle one can jsut use the LogicalErrorRates array
    Data = {"LogicalErrorRates" : LogicalErrorRates,"InitialLogicalErrorRates":InitialLogicalErrorRates,"FinalLogicalErrorRates":FinalLogicalErrorRates,"Parameters":Parameters.ToDict(),"RelativeImprovements":RelativeImprovements, "RealPhysicalRates": RealPhysicalRates, "EstimatedPhysicalRates": EstimatedPhysicalRates, "RealMeasurementErrorRates":RealMeasurementRates, "EstimatedMeasurementErrorRates": EstimatedMeasurementRates}
    with open(SaveFolder+"/"+name+".pkl","wb") as out:
        pickle.dump(Data,out)
    with open(SaveFolder+"/"+name+".info","w") as out:
        json.dump(Parameters.ToDict(),out)
        out.write("\n")
        out.write("# Average Relative Improvement: " + str(AverageRelativeImprovement)) 
        out.write("\n"+"#"+str(datetime.datetime.now()))    

def Setup5QubitModels(n_concat,p_mean,p_mean_measurement,Pseudocounts = 20,logdomain = True,FixRealRates = False):
    """
    Generates two 5 qubit models, one with the Real Rates drawn from a dirichlet around p_mean, and one with fixed Guess Rates of p_mean
    If FixRealRates = True, the roles of are exchanged, i.e. the real rates are fixed to p_mean and the guess is drawn from the dirichlet
    """
    BaseCode = bp.Create5QubitPerfectCode()
    n_physical_qubits = BaseCode.n_qubits ** n_concat
    n_total_checks = n_physical_qubits - 1 #Assumes only one qubit is encoded
    RealRates = GetRandomRates(p_mean,Pseudocounts,n_physical_qubits,cardinality=4)
    GuessRates = np.reshape(np.array([1-3*p_mean,p_mean,p_mean,p_mean]*n_physical_qubits),(-1,4))
    if p_mean_measurement is not None:
        RealMeasRates = GetRandomRates(p_mean_measurement,Pseudocounts,n_total_checks,cardinality=2)
        GuessMeasRates = [1-p_mean_measurement,p_mean_measurement]*n_total_checks
    else:
        RealMeasRates = None
        GuessMeasRates = None
    RealModel = bp.ConcatQCode(BaseCode, n_concat,RealRates,RealMeasRates,logdomain)
    GuessModel = bp.ConcatQCode(BaseCode, n_concat,GuessRates,GuessMeasRates,logdomain)
    if FixRealRates == False:
        return RealModel,GuessModel
    else:
        return GuessModel,RealModel
def GetRandomRates(p_mean,PseudoSamples,n,cardinality):
    #alpha = [PseudoSamples*(1-3*p_mean),PseudoSamples*p_mean,PseudoSamples*p_mean,PseudoSamples*p_mean]
    c = cardinality-1
    alpha = [PseudoSamples*(1-c*p_mean)] + [PseudoSamples*p_mean]*c
    ErrorRates = np.random.dirichlet(alpha,n)    
    return ErrorRates