import itertools
import math
import numpy as np

from numba import jit
import numba

"""
Belief Propagation for Concatenated Codes with Single qubit Errors, c.f. "[Poulin]Optimal And Efficient Decoding of Quantum Block Codes"
Only supports independent single qubit Pauli errors as error model.
Also includes measurement Errors
"""
"""
Formats
Pauli Strings: As quaternary vectors (Id,X,Z,Y = 0,1,2,3)
"""
@jit(nopython = True)
def NumbaCategorical(ErrorRates,n_repeat):
    """
    Samples n_repeat Pauli Strings P_i of length ErrorRates.shape[0] from Categorical distributions, where the distribution of the Pauli P_i_j is Categorical specified by ErrorRates[i,:]
    """
    Randoms = np.random.rand(n_repeat,ErrorRates.shape[0])
    Errors = np.empty((n_repeat,ErrorRates.shape[0]),dtype=numba.int64)
    #Errors = np.empty((n_repeat,ErrorRates.shape[0]),dtype=int)
    for n in range(n_repeat):
        for i in range(ErrorRates.shape[0]):
            Rates = ErrorRates[i]
            r = Randoms[n,i]
            Errors[n,i] = RealToCategorical(Rates, r)
    return Errors
@jit(nopython = True)
def RealToCategorical(Probabilities, r):
    """
    Helper for NumbaCategorical
    Converts a real number to a categorical variable (with discrete values) according to a probability table
    """
    s = 0
    for i in range(Probabilities.size):
        s += Probabilities[i]
        if r < s:
            return i    

def AssignmentIterator(Cardinalities):
    """
    Returns an iterator over all possible assignments with these cardinalities
    """
    a = [np.arange(0,c,dtype = int) for c in Cardinalities]
    return itertools.product(*a)
@jit(nopython = True)
def AssignmentToIndex(Assignment,cardinalities):
    """
    Compute the Index associated with an assignment according to libDAI convention
    """
    y = 0
    c = 1
    for i in range(Assignment.size):
        y += c*Assignment[i]
        c *= cardinalities[i]
    return int(y)
@jit(nopython = True)
def SyndromeToIndex(Assignment):
    """
    Compute the Index associated with an assignment according to libDAI convention
    """
    y = 0
    c = 1
    for i in range(Assignment.size):
        y += c*Assignment[i]
        c *= 2
    return int(y)
    
@jit(nopython = True)
def QuaternaryProduct(E1,E2):
    """
    Returns 1 if the Pauli Strings anti-commute and 0 otherwise
    """
    Sum = 0
    for i in range(E1.shape[0]):
        Sum += ScalarQuaternaryProduct(E1[i],E2[i])
    return Sum % 2
@jit(nopython = True)
def ScalarQuaternaryProduct(x,y):
    """
    Helper for Quaternary Product
    """
    if x == 0 or y == 0:
        return 0
    else:
        return not x == y
def QuaternaryAddition(E1,E2):
    """
    Returns the Product of the Pauli Strings, i.e. the Sum of their quaternary representations
    """
    S1 = QuaternaryToSymplectic(E1)
    S2 = QuaternaryToSymplectic(E2)
    R = (S1 + S2) % 2
    R = SymplecticToQuaternary(R)
    return R

"""
Methods to switch between Quaternary and Symplectic Representation of Pauli strings
"""
def QuaternaryToSymplectic(E):
    E = np.array(E)
    n_dim = E.ndim
    E_vec = E
    if n_dim == 1:
        E_vec = np.array([E])
    E_Sym  = np.zeros((E_vec.shape[0],2*E_vec.shape[1]))
    for i,e in enumerate(E_vec):
        E_Sym[i,:] = _QuaternaryToSymplectic(e)
    if n_dim == 1:
        E_Sym = E_Sym[0]
    return E_Sym
def _QuaternaryToSymplectic(e):
    e = np.array(e)
    n = e.size
    e_new = np.zeros(n * 2)
    for i,v in enumerate(e):
        if v == 0:
            e_new[i] = 0
            e_new[i+n] = 0
        elif v == 1:
            e_new[i] = 1
            e_new[i+n] = 0
        elif v == 2:
            e_new[i] = 0
            e_new[i+n] = 1
        elif v == 3:
            e_new[i] = 1
            e_new[i+n] = 1
        else:
            raise ValueError("Input was not quaternary")
    return e_new
def SymplecticToQuaternary(E):
    n_dim = E.ndim
    E_vec = E
    if n_dim == 1:
        E_vec = np.array([E])
    n = int(E_vec.shape[1]/2)
    E_Q  = np.zeros((E_vec.shape[0],n))
    for i,e in enumerate(E_vec):
        E_Q[i,:] = _SymplecticToQuaternary(e)
    if n_dim == 1:
        E_Q = E_Q[0]
    return E_Q
def _SymplecticToQuaternary(e):
    n = int(e.size/2)
    e_new = np.zeros(n)
    for i in range(n):
        x = e[i]
        z = e[n+i]
        if x == 0:
            if z == 1:
                e_new[i] = 1 #Z
            else:
                e_new[i] = 0 #ID
        else:
            if z == 1:
                e_new[i] = 3 #Y
            else:
                e_new[i] = 2 #X
    return e_new



class QCode:
    """
    Class representing a Quantum Code as a Factor Graph
    Stabilizers and Logical Operators should be given as Quaternary arrays
    Canonical order of Variables here: Qubit Variables| Syndrome Variables | Log Op Variables
    It is (currently) assumed that the code encodes only one logical qubit. Therefore there should be two logical operators. The X operator should be given first.
    """
    def __init__(self,Stabilizers,LogicalOperators,name = ""):
        #Stabilizers as Quaternary Array, where each row is a Stabilizer operator
        self.Stabilizers = np.array(Stabilizers,dtype = int)
        #The Logical X and Z Operator of the Code
        self.LogicalOperators = np.array(LogicalOperators,dtype = int)
        #The number of Qubits
        self.n_qubits = self.Stabilizers.shape[1]  
        #The number of syndrome bits
        self.n_checks = self.Stabilizers.shape[0]
        #The Cardinalities associated with Qubits and Detections, i.e. 4 for qubit and 2 for syndrome bit
        self.QubitCard = 4
        self.CheckCard = 2
        #A name for the code
        self.name = name
        #All Possible Errors on the Code
        self.Errors = np.zeros((self.QubitCard**self.n_qubits,self.n_qubits),dtype = int)
        #A LookUp Table of the syndrome for each error, in the same order as the errors
        self.SyndromeLUT = np.zeros((self.QubitCard**self.n_qubits,self.n_checks),dtype = bool)
        #A LookUp Table of the logical error for each error, in the same order as the errors
        self.LogicalLUT = np.zeros(self.QubitCard**self.n_qubits,dtype = int)
        
        #A Table giving the matching errors for each combination of syndromes and logical error
        n_syns = int(2**self.n_checks)
        n_err_per_syn = int(4**self.n_qubits / n_syns)
        self.ErrorsPerSyndrome = np.zeros((n_syns,n_err_per_syn,self.n_qubits+1),dtype = int)
        self.MeasurementErrors = np.array(list(AssignmentIterator([2]*self.n_checks)),dtype=int) #List of possible measurement errors = List of binary strings of length n_checks
        
        self._initialize()
    
    def _initialize(self):
        for i,e in enumerate(AssignmentIterator([self.QubitCard]*self.n_qubits)):
            error = np.array(e,dtype = int)
            self.Errors[i,:] = error
            self.SyndromeLUT[i,:] = self.ComputeSyndrome(error)
            self.LogicalLUT[i] = self.ComputeLogicalError(error)
            shape = tuple([self.QubitCard]*self.n_qubits)
        #The lookuptables in reshaped form such that SyndromeLUTr[e] = syn(e) for any error e (as a tuple)
        self.SyndromeLutr = np.reshape(self.SyndromeLUT,shape+(self.n_checks,))
        self.LogicalLutr = np.reshape(self.LogicalLUT,shape)
        #Lookuptable with a list of matching errors for each syndrome and logical error
        Indizes = np.zeros(self.ErrorsPerSyndrome.shape[0],dtype = int)
        for e in self.Errors:
            syn,l = self.GetSyndromeAndLogicalError(e)
            ind = SyndromeToIndex(syn)
            self.ErrorsPerSyndrome[ind,Indizes[ind]] = np.append(e, l)
            Indizes[ind] += 1

    def ComputeLogicalError(self,error):
        """
        Computes the logical error of a the physical error "error"
        """
        lz = QuaternaryProduct(error,self.LogicalOperators[0]) #A logical Z error anti commutes with the logical X
        lx = QuaternaryProduct(error,self.LogicalOperators[1]) # A logical X error anti commutes with the logical Z
        v = SymplecticToQuaternary(np.array([lz,lx]))[0] #Quaternary value indicating the Logical Error
        return v

    def ComputeSyndrome(self,error):
        """
        Computes the syndrome of a the physical error "error"
        """
        error = np.array(error,dtype = int)
        return self._NumbaComputeSyndrome(error,self.Stabilizers)
    @staticmethod
    @jit(nopython = True)
    def _NumbaComputeSyndrome(error,Stabilizers):
        Syndrome = np.zeros(Stabilizers.shape[0])
        for i in range(Stabilizers.shape[0]):
            s = Stabilizers[i]
            p = QuaternaryProduct(error,s)
            Syndrome[i] = p
        return Syndrome  
    
    def GetSyndromeAndLogicalError(self,error):
        """
        Looks up the syndrome and Logical Error in LookUp Tables, cannot be used for initialization, but after initialization faster than direct computation
        """
        e = tuple(error.astype(int))
        L = self.LogicalLutr[e]
        syn = self.SyndromeLutr[e]
        return syn,L
    
def Create5QubitPerfectCode():
    """
    The 5 Qubit Perfect Code
    """
    Stabilizers = np.array([[1,2,2,1,0],
                            [0,1,2,2,1],
                            [1,0,1,2,2],
                            [2,1,0,1,2]],dtype = int)
    LogX = np.array([1,1,1,1,1])
    LogZ = np.array([2,2,2,2,2])
    LogOps = np.vstack((LogX,LogZ))
    Code = QCode(Stabilizers,LogOps,"5QubitPerfectCode")
    return Code
def Create3QubitRepCode():
    """
    The 3 Qubit Rep Code
    """
    Stabilizers = np.array([[2,2,0],[0,2,2]],dtype = int)
    LogX = np.array([1,1,1])
    LogZ = np.array([2,2,2])
    LogOps=np.vstack((LogX,LogZ))
    Code = QCode(Stabilizers,LogOps,"3QubitRepCode")
    return Code


class ConcatQCode:
    """
    A quantum code concatenated with itself a number of times, and the associated tree structure for belief propagation
    Also includes measurement errors
    """    
    def __init__(self,BaseCode,n_concatenations,ErrorRates = None,MeasurementErrorRates = None, logdomain = True):
        """
        If MeasurementErrorRates is None, no nodes are set up for measurement errors to safe computation time in BP. Thus this is saves computation time vs setting the measurement error rates to 0.
        """
        self.BaseCode = BaseCode
        self.n_concatenations = n_concatenations
        self.n_block = self.BaseCode.n_qubits
        self.n_check_block = self.BaseCode.n_checks
        self.QubitLayers = [] #A list of all (logical) QubitNodes. Each entry is one layer of nodes, starting at the leaves.
        self.MeasurementErrorLayers = [] #A list of all measurement error nodes. Each entry is one layer of measurement errors, starting with the lowest layer at the bottom syndrome measurements.
        self.FactorLayers = [] #All the factors except the ones for measurement errors
        self.n_physical_Qubits = self.BaseCode.n_qubits ** n_concatenations
        self.n_total_Qubits = int((self.n_block**(self.n_concatenations+1) - 1) / (self.n_block - 1)) #See Wikipedia m-ary tree
        #Syndromes Per Qubit
        self.ChecksPerQubit = self.n_check_block / self.n_block
        #Total nr Checks
        self.n_total_checks = int(self.n_total_Qubits * self.ChecksPerQubit)
        self.HasMeasurementErrors = False
        
        if ErrorRates is None:
            self.ErrorRates = np.reshape(np.array([1,0,0,0]*self.n_physical_Qubits), (self.n_physical_Qubits,4)) #Assumes quaternary representation
        else:
            self.ErrorRates = np.reshape(ErrorRates, ((self.n_physical_Qubits,4)))
        if MeasurementErrorRates is not None:
            self.MeasurementErrorRates = self._ReshapeMeasurementErrorRates(MeasurementErrorRates)
            self.HasMeasurementErrors = True
        else:
            self.MeasurementErrorRates = None
        self.SetupNodes(self.ErrorRates,self.MeasurementErrorRates,logdomain)
        self.logdomain = logdomain
    def _ReshapeMeasurementErrorRates(self,MeasurementErrorRates):
        """
        Reshapes measurement error rates into a format where they are specified layer wise
        """
        #MeasurementErrorRates = np.concatenate(MeasurementErrorRates) #Flatten if it was given as list of layers
        if len(MeasurementErrorRates) == self.n_concatenations:
            #In this case the rates are already in layer structure
            return MeasurementErrorRates
        MeasurementErrorRates = np.reshape(MeasurementErrorRates, (self.n_total_checks,2))
        MeasurementErrorRatesNew = []
        c = 0
        for i in range(self.n_concatenations):
            n_qubits_layer = int((self.BaseCode.n_qubits ** (self.n_concatenations - i))*self.ChecksPerQubit)
            MeasurementErrorRatesLayer = MeasurementErrorRates[c:c+n_qubits_layer]
            c += n_qubits_layer
            MeasurementErrorRatesNew += [MeasurementErrorRatesLayer]
        MeasurementErrorRatesNew=np.array(MeasurementErrorRatesNew)
        return MeasurementErrorRatesNew
    
    def GetRates(self,flat = True):
        """
        Return error rates and measurement error rates either layer-wise or as a flat array
        """
        e = np.array(self.ErrorRates)
        m = self.MeasurementErrorRates
        if flat == True:
            if self.MeasurementErrorRates is not None:
                m = np.concatenate(m) #Flatten the rates into an array of shape (n_total_checks,2) instead of Layer structure
                m = np.array(m)
        return e,m
            
    def SetupNodes(self,ErrorRates,MeasurementErrorRates,logdomain):
        #Setup the leaves, i.e. the physical qubits (and the first layer of measurement errors, below)
        PhysicalQubits = [QubitVariableNode(name = "q_0_"+str(i),NodeType = 0,logdomain = logdomain) for i in range(self.BaseCode.n_qubits ** self.n_concatenations)]        
        #Setup the factor graph of the code
        self.QubitLayers += [PhysicalQubits]
        if self.HasMeasurementErrors == True:
            FirstMeasurementErrors = [QubitVariableNode(name = "m_0_"+str(i),NodeType = 1,logdomain=logdomain) for i in range(int(self.BaseCode.n_qubits ** self.n_concatenations * self.ChecksPerQubit))]
            self.MeasurementErrorLayers += [FirstMeasurementErrors]
        for i in range(self.n_concatenations):
            
            #The factors connecting to the next layer
            n_factors = self.BaseCode.n_qubits ** (self.n_concatenations - i- 1)
            FactorLayer = [LogicalFactorNode(self.BaseCode, name = "f_"+str(i)+"_"+str(j),logdomain = logdomain) for j in range(n_factors)]
            for j,f in enumerate(FactorLayer):
                f.children = self.QubitLayers[i][self.BaseCode.n_qubits*j:self.BaseCode.n_qubits*(j+1)] #The qubit nodes connected
                if self.HasMeasurementErrors == True:
                    f.children += self.MeasurementErrorLayers[i][self.BaseCode.n_checks*j:self.BaseCode.n_checks*(j+1)] # The connected measurement errors
                for c in f.children:
                    c.parent = f
           
            #The qubits of the next layer
            n_LogicalQubits = n_factors
            LogicalQubitLayer = [QubitVariableNode(name = "q_"+str(i+1)+"_"+str(j),NodeType = 0,logdomain = logdomain) for j in range(n_LogicalQubits)]
            for j,q in enumerate(LogicalQubitLayer):
                q.children = [FactorLayer[j]]
                for f in q.children:
                    f.parent = q
                    
            #The measurement errors of the next layer
            if self.HasMeasurementErrors == True:
                n_MeasurementErrors = int(n_LogicalQubits * self.ChecksPerQubit)            
                MeasurementErrorLayer = [QubitVariableNode(name = "m_"+str(i+1)+"_"+str(j),NodeType = 1,logdomain=logdomain) for j in range(n_MeasurementErrors)]
                #Note that parents are added in the next iteration, and the only child will be an error factor node giving the error rate added later
                
            self.FactorLayers += [FactorLayer]
            self.QubitLayers += [LogicalQubitLayer]
            if self.HasMeasurementErrors == True:
                self.MeasurementErrorLayers += [MeasurementErrorLayer]
        
        self.SetupErrorFactors(ErrorRates,MeasurementErrorRates,logdomain)
        
        for layer in self.FactorLayers + self.QubitLayers + self.MeasurementErrorLayers:
            for n in layer:
                n.InitMessages()
       
    def SetupErrorFactors(self,ErrorRates,MeasurementErrorRates,logdomain):
        #Setup the factors for the error rates
        ErrorLayer = []
        ErrorRates = np.reshape(ErrorRates,(self.n_physical_Qubits,4)) #Assumes quaternary representation
        for i,q in enumerate(self.QubitLayers[0]):
            f = ErrorFactorNode(ErrorRates[i], parent = q,name = "e_"+str(i),logdomain = logdomain)
            q.children = [f]
            ErrorLayer += [f]
        self.FactorLayers = [ErrorLayer] + self.FactorLayers
        #Setup factors for measurement error rates
        if self.HasMeasurementErrors == True:
            for i,l in enumerate(self.MeasurementErrorLayers):
                MeasurementErrorFactorLayer = []
                for j,m in enumerate(l):
                    f = ErrorFactorNode(MeasurementErrorRates[i][j],parent=m,name="em_"+str(i)+"_"+str(j),logdomain=logdomain)
                    m.children=[f]
                    MeasurementErrorFactorLayer += [f]
                self.FactorLayers[0] += MeasurementErrorFactorLayer #We store all the measurement errors factors, irrespective of actual layer, together with the error factors, because they are all leaf nodes and should pass messages in BP according to the schedule for leaf nodes
               
    def SetErrorRates(self,ErrorRates=None,MeasurementErrorRates = None):
        """
        Can either pass Qubit Error Rates and Measurement Error Rates as separate arguments or Pass only the ErrorRates argument as list [QubitErrorRate_1,...,QubitErrorRate_n,MeasurementErrorRate_1,...,MeasurementErrorRate_m] where QubitErrorRate_i = [p_i,p_x,p_z,p_y] and MeasurementErrorRate_i = [p_0,p_1]
        Cannot set Measurement Error Rates if the code did not have measurement errors in the beginning.
        """
        if ErrorRates is not None:
            if ErrorRates.size == 4*self.n_physical_Qubits:
                """
                In this case it is a list of qubit error rates
                """
                self._SetErrorRates(ErrorRates,MeasurementErrorRates)
            elif ErrorRates.size == (4*self.n_physical_Qubits + 2*self.n_total_checks):
                """
                In this case it is a list of both qubit and measurement error rates
                """
                QubitErrorRates = ErrorRates[0:self.n_physical_Qubits]
                MeasurementErrorRates = ErrorRates[self.n_physical_Qubits:]
                self._SetErrorRates(QubitErrorRates,MeasurementErrorRates)
            elif ErrorRates.size == (2*self.n_total_checks):
                """
                In this case its a list of only measurement error rates
                """
                self._SetErrorRates(None,MeasurementErrorRates)
            else:
                raise ValueError("Shape of Error Rates does not match code")
        else:
            self._SetErrorRates(ErrorRates,MeasurementErrorRates)
    

    def _SetErrorRates(self,QubitErrorRates=None,MeasurementErrorRates=None):
        """
        Cannot set Measurement Error Rates if the code did not have measurement errors in the beginning.
        Assumes that Error Layer was Setup
        """
        if QubitErrorRates is not None:
            self.ErrorRates = np.reshape(QubitErrorRates, ((self.n_physical_Qubits,4)))
            for i in range(self.n_physical_Qubits):
                f = self.FactorLayers[0][i]
                f.SetErrorRates(self.ErrorRates[i])
        if self.HasMeasurementErrors:
            if MeasurementErrorRates is not None:
                self.MeasurementErrorRates = self._ReshapeMeasurementErrorRates(MeasurementErrorRates)
                MeasRates = self.GetRates(flat=True)[1] #Flatten the list of layers into a numpy array of the eror rates
                for i in range(self.n_total_checks):
                    f = self.FactorLayers[0][self.n_physical_Qubits + i]
                    f.SetErrorRates(MeasRates[i])
            
    def PrintLayers(self):
        """
        Prints the parents and childrens of each layer for debugging
        """
        for i,layer in enumerate(self.QubitLayers + self.FactorLayers + self.MeasurementErrorLayers):
            print()
            print("Layer")
            for n in layer:
                node = n.name
                children = [c.name for c in n.children]
                if n.parent is not None:
                    parent = n.parent.name
                else:
                    parent = ""
                print("Node: ", node ," Children: ",  children ," Parent:" , parent)
    
    def PrintMessages(self):
        for l in self.QubitLayers + self.MeasurementErrorLayers + self.FactorLayers:
            for f in l:
                print(f.name)
                print(f.messages)
                print()
                
    def SetLogdomain(self,logdomain):
        """
        Decide whether message passing should be done in the log-domain or not. Log-domain is numerically more stable and also a bit faster.
        """
        for layer in self.FactorLayers:
            for f in layer:
                f.SetLogdomain(logdomain)
        for layer in self.QubitLayers:
            for q in layer:
                q.SetLogdomain(logdomain)
        if self.HasMeasurementErrors == True:
            for layer in self.MeasurementErrorLayers:
                for m in layer:
                    m.SetLogdomain(logdomain)
        self.logdomain = logdomain
    
    def BPUpwardsPass(self,Syndrome):
        """
         The upwards pass of belief propagation from leafs to root. Assumes leaf nodes are initialized.
         Syndrome should be given starting from the bottom layer, row wise left to right.
        """
        Syndrome = np.reshape(Syndrome,(-1,self.BaseCode.n_checks))
        ind = 0
        for i in range(self.n_concatenations):
            for q in self.QubitLayers[i]:
                q.UpdateMessageToParent()
            if self.HasMeasurementErrors == True:
                for m in self.MeasurementErrorLayers[i]:
                    m.UpdateMessageToParent()
            for f in self.FactorLayers[i+1]: #Note that the error factors never have to update the message
                f.UpdateMessageToParent(Syndrome[ind])
                ind += 1

    def BPDownwardsPass(self,Syndrome):
        """
         The downwards pass of belief propagation from root to leafs. Assumes upward pass was done before.
         Syndrome should be given starting from the bottom layer, row wise left to right.
        """
        Syndrome = np.reshape(Syndrome,(-1,self.BaseCode.n_checks))
        ReversedSyndrome = list(reversed(Syndrome))
        FactorLayersReversed = list(reversed(self.FactorLayers))
        QubitLayersReversed = list(reversed(self.QubitLayers))
        MeasurementErrorLayersReversed = list(reversed(self.MeasurementErrorLayers))
        ind = 0
        for i in range(self.n_concatenations):
            for q in QubitLayersReversed[i]:
                q.UpdateMessagesToChildren()
            if self.HasMeasurementErrors == True:
                for m in MeasurementErrorLayersReversed[i]:
                    m.UpdateMessagesToChildren()
            for f in reversed(FactorLayersReversed[i]): #Again the Error Layer doesnt have to update
                f.UpdateMessagesToChildren(ReversedSyndrome[ind])
                ind += 1

    def RunBP(self,Syndrome):
        """
        Runs the Belief Propagation algorithm to determine marginals as explained in e.g. Bishop:Pattern Recognition and machine learning or Koller: Probabilistic Graphical Models and Techniques
        """
        self.BPUpwardsPass(Syndrome)
        self.BPDownwardsPass(Syndrome)
        
    def LeafMarginals(self):
        """
        Compute the Marginal Distribution of each leaf, i.e. for each qubit and measurement error
        Assumes that RunBP() was called beforehands
        """
        Marginals = [c.CalculateMarginal() for c in self.QubitLayers[0]]
        if self.HasMeasurementErrors:
            Marginals_m = [c.CalculateMarginal() for  l in self.MeasurementErrorLayers for c in l]
            Marginals += Marginals_m
        return Marginals
    def TopMarginal(self):
        """
        Computes the distribution of logical errors, i.e. the marginal of the root node
        Assumes that RunBP() was called beforehands
        """
        Marginal = self.QubitLayers[-1][0].CalculateMarginal()
        return Marginal   
    
    def GenerateData(self,n):
        """
        Generates a Set of Syndromes and Corresponding Logical Errors according to the distribution given by the error rates
        """        
        Syndromes = np.zeros((n,self.n_total_checks),dtype = bool)
        LogicalErrors = np.zeros((n),dtype = int)
        PhysicalErrors = NumbaCategorical(self.ErrorRates, n)
        for i in range(n):
            e = PhysicalErrors[i]
            s,l = self.ComputeSyndromeAndLogicalError(e)
            Syndromes[i] = s
            LogicalErrors[i] = l
        if self.HasMeasurementErrors:
            #MRates =np.concatenate(self.MeasurementErrorRates) #Flatten the list into an numpy array
            MRates= self.GetRates(flat=True)[1]
            MeasurementErrors = NumbaCategorical(MRates,n)
            Syndromes = (Syndromes + MeasurementErrors) % 2
        return Syndromes,LogicalErrors,PhysicalErrors
    def GenerateDatum(self):
        d = self.GenerateData(1)
        s = d[0][0]
        l = d[1][0]
        p = d[2][0]
        return s,l,p
     
    def MaxSumUpwardsPass(self,Syndrome):
        """
         The upwards pass of the max sum algorithm from leafs to root. Assumes leaf nodes are initialized.
         Syndrome should be given starting from the bottom layer, row wise left to right.
        """
        Syndrome = np.reshape(Syndrome,(-1,self.BaseCode.n_checks))
        ind = 0
        for i in range(self.n_concatenations):
            for q in self.QubitLayers[i]:
                q.UpdateMaxSumMessageToParent()
            if self.HasMeasurementErrors:
                for m in self.MeasurementErrorLayers[i]:
                    m.UpdateMaxSumMessageToParent()
            for f in self.FactorLayers[i+1]: #Note that the error factors never have to update the message
                f.UpdateMaxSumMessageToParent(Syndrome[ind])
                ind += 1 
        #Update the top node
        self.QubitLayers[-1][0].UpdateMaxSumMessageToParent()
    def MaxSumDownwardsPass(self):
         #Update the state of the top nodes
        self.QubitLayers[-1][0].UpdateMaxSumState()
        #Do the remaining states by backtracking
        FactorLayersReversed = list(reversed(self.FactorLayers))
        for Layer in FactorLayersReversed[:-1]: #Dont update the Error Layer
            for f in Layer:
                f.UpdateChildrenMaxSum()
    def RunMaxSum(self,Syndrome):
        """
        Runs the Max Sum algorithm to determine marginals as explained in e.g. Bishop:Pattern Recognition and machine learning
        """
        self.MaxSumUpwardsPass(Syndrome)
        self.MaxSumDownwardsPass()
    def MAPConfig(self):
        """
        Returns the most likely error. (Including measurement errors if applicable)
        Assumes that RunMaxSum was called beforehands.
        """
        e = [n.state for n in self.QubitLayers[0]]
        if self.HasMeasurementErrors:
            m = [n.state for l in self.MeasurementErrorLayers for n in l]
        else:
            m = []
        e = np.array(e,dtype=int)
        m = np.array(m,dtype=int)
        config = np.concatenate((e,m))
        return config
        
    def ComputeSyndromeAndLogicalError(self,error,measurementerror = None):
        """
        Compute Syndrome and Logical Error of a given Physical Error
        Does not inherently take into account measurement errors
        """
        self.SetQubitStates(error)
        Syndrome = np.reshape(np.zeros(self.n_total_checks,dtype=int),(-1,self.n_check_block))
        index = 0
        for layer in self.FactorLayers[1:]: #FactorLayers except the error layer
            for f in layer:
                Syndrome[index] = f.PropagateError()
                index += 1
        LogicalError = self.QubitLayers[-1][0].state #State of the top node is logical error
        Syndrome=Syndrome.flatten()
        if measurementerror is not None:
            Syndrome = (Syndrome + measurementerror) % 2
        return Syndrome.flatten(), LogicalError
    def SetQubitStates(self,error,measurementerror=None):
        for i,q in enumerate(self.QubitLayers[0]):
            q.state = error[i]
        if measurementerror is not None:
            if self.HasMeasurementErrors:
                c = 0
                for l in self.MeasurementErrorLayers:
                    for m in self.layer:
                        m.state=measurementerror[c]
                        c+=1

    def CalculateProb(self,e,m=None,log=True):
        """
        Calculates the probability of the given error given the current error rates
        If log = True the log of the probability is returned, numerically more stable
        e = error on qubits
        m = measurement error
        """
        e = np.array(e,dtype=int)
        if m is not None:
            m = np.array(m,dtype=int)
        ErrorRates,MeasurementErrorRates = self.GetRates(flat=True)
        if self.HasMeasurementErrors and m is not None:
            prob=self._CalculateProbM(e,ErrorRates,m,MeasurementErrorRates,log)
        else:
            prob=self._CalculateProb(e,ErrorRates,log)   
        return prob
    @staticmethod
    @jit(nopython=True)
    def _CalculateProbM(e,ErrorRates,m,MeasurementErrorRates,log):
        if log==True:
            ErrorRates = np.log(ErrorRates)
            MeasurementErrorRates = np.log(MeasurementErrorRates)
            p = 0
        else:
            p = 1
        for i in range(e.size):
            if log == True:
                p += ErrorRates[i][e[i]]
            else:
                p *= ErrorRates[i][e[i]]
        for i in range(m.size):
            if log == True:
                p += MeasurementErrorRates[i][m[i]]
            else:
                p *= MeasurementErrorRates[i][m[i]]
        return p
    @staticmethod
    @jit(nopython=True)
    def _CalculateProb(e,ErrorRates,log):
        if log==True:
            ErrorRates = np.log(ErrorRates)
            p = 0
        else:
            p = 1
        for i in range(e.size):
            if log:
                p += ErrorRates[i][e[i]]
            else:
                p *= ErrorRates[i][e[i]]
        return p
    
        

class Node:
    """
    A node in the tree of a Concat QCode
    Saves messages to parents and children for BP, as well as messages and a possibly a state for max_sum
    """
    def __init__(self,name = "",logdomain = True):
        self.children = []
        self.parent = None
        self.messages = [] #Messages for Belief propagation
                           #One message for each neighbour, each message is a vector with one entry for each possible state of the neighbour
                           #Order of messages is children in their order then parent
        self.maxsummessage = None #The message to parent for max sum (see e.g. [Bishop,p.413])
        self.name = name #Mainly useful for debugging
        self.logdomain = logdomain #Whether Belief Propagation messages should be done in log domain for more numerical stability
    
    def InitMessages(self,Messages = None):
        """
        Assumes that parent and children have been set
        """
        if Messages is None:
            self.messages = [np.zeros(4) for n in self.children + [self.parent]]
        elif Messages.ndim == 1: #Convenience for uniform initialization
            self.messages = np.array([Messages for i in range(len(self.children)+1)])
        
        elif Messages.shape[0] == len(self.children)+1: #Check for correct number of messages
            self.messages = Messages
        else:
            raise ValueError("Message number  is incorrect")
    
    def GetMessage(self,TargetNode):
        """
        Get meassage incoming to this node from TargetNode (TargetNode should be the parent or a child)
        """
        if TargetNode is self.parent:
            return self.messages[-1]
        else:
            index = self.children.index(TargetNode)
            return self.messages[index]
    def GetMaxSumMessage(self):
        return self.maxsummessage
        
    def Neighbors(self):
        """
        Returns parent and children
        """
        nodes = self.children
        if self.parent is not None:
            nodes = nodes + [self.parent]
        return nodes
        
    def CalculateMessage(TargetNode):
        raise NotImplementedError("Message Calculation only implemented in Derived Classes")
    def SetLogdomain(self,logdomain):
        self.logdomain = logdomain
    def PrintChildren(self):
        for c in self.children:
            print(c.name)
                
class QubitVariableNode(Node):
    """
    Used to represent (logical) Qubit Variables and Measurement Error Variables. Saves a state for max_sum and syndrome computation, representing the error on the node
    """
    def __init__(self,name = "",NodeType=0,logdomain = True):
        Node.__init__(self,name,logdomain)
        self.NodeType = NodeType # 1 for measurement error nodes, 0 for qubit variable nodes
        if self.NodeType == 0:
            self.n_states = 4      #4 states for qubit nodes
        else:
            self.n_states = 2       #only 2 states for measurement error nodes
        self.state = 0 #Used in syndrome Computation and max sum, not Belief Propagation
        
    def InitMessages(self,Messages = None):
        if Messages is None:
            self.messages = [np.zeros(self.n_states) for n in self.children + [self.parent]] 
        elif Messages.ndim == 1: #Convenience for uniform initialization
            self.messages = np.array([Messages for i in range(len(self.children)+1)])
        
        elif Messages.shape[0] == len(self.children)+1: #Check for correct number of messages
            self.messages = Messages
        else:
            raise ValueError("Message number  is incorrect")
            
    def UpdateMessage(self,TargetNode):
        """
        Update for Belief Propagation
        """
        message = self.CalculateMessage(TargetNode)
        if TargetNode is self.parent:
            self.messages[-1] = message
        else:
            index = self.children.index(TargetNode)
            self.messages[index] = message
    def UpdateMessageToParent(self,logdomain = True):
        if self.parent is not None:
            self.UpdateMessage(self.parent)            
    def UpdateMessagesToChildren(self,logdomain = True):
        for c in self.children:
            self.UpdateMessage(c)    
    def CalculateMessage(self,TargetNode):
        """
        Calculates message for Belief Propagation
        """
        if (TargetNode not in self.children) and (TargetNode is not self.parent):
            raise ValueError("TargetNode is not connected to Node")
        Messages = [n.GetMessage(self) for n in self.Neighbors() if n is not TargetNode]
        Messages = np.array(Messages)
        result = self._CalculateMessage(Messages,self.logdomain,self.n_states)
        return result
    @staticmethod
    @jit(nopython= True)
    def _CalculateMessage(Messages,logdomain,n_states):
        if logdomain:
            result = np.zeros(n_states)
        else:
            result = np.ones(n_states)
        for i in range(Messages.shape[0]):
                if logdomain:
                    result += Messages[i]
                else:
                    result *= Messages[i]
        return result
           
    def CalculateMarginal(self):
        Messages = np.array([node.GetMessage(self) for node in self.Neighbors()])
        return self._NumbaCalculateMarginal(Messages,self.logdomain)
    @staticmethod
    @jit(nopython = True)
    def _NumbaCalculateMarginal(Messages,logdomain):
        if logdomain:
            result = np.zeros(Messages.shape[1]) #Assumes that messages are for the correct number of states
        else:
            result = np.ones(Messages.shape[1])
        for i in range(Messages.shape[0]):
                if logdomain:
                    result += Messages[i]
                else:
                    result *= Messages[i]
        if logdomain:
            result = np.exp(result)
        result /= np.sum(result)
        return result

    def UpdateMaxSumMessageToParent(self):
        message = self.CalculateMaxSumMessageToParent()
        self.maxsummessage = message
    def CalculateMaxSumMessageToParent(self):
        Messages = [n.GetMaxSumMessage() for n in self.children]
        Messages = np.array(Messages)
        result = self._CalculateMaxSumMessageToParent(Messages,self.n_states)
        return result
    @staticmethod
    @jit(nopython=True)
    def _CalculateMaxSumMessageToParent(Messages,n_states):
        result = np.zeros(n_states)
        for i in range(Messages.shape[0]):
            result += Messages[i]
        return result      
    def UpdateMaxSumState(self):
        """
        Sets the max sum state according to the messages from its children. 
        Only used by the top node in max sum algorithm, other node states are determined by by backtracking
        Assumes that UpdateMaxSumMessageToParent() was called before
        """
        self.state = np.argmax(self.maxsummessage)
            
class ErrorFactorNode(Node):
    """
    Used to represent the error rates for Qubit or Measurement errors (quaternary state for qubit, binary for measurement errors)
    """
    def __init__(self,ErrorRates,parent,name = "",logdomain = True):
        """
        Error Rates should be the Pauli I,X,Z,Y rates of the associated qubit -> 4 elements
        Or for measurement errors, the probability of No error / error -> 2 elements
        """
        Node.__init__(self,name,logdomain)
        self.parent = parent
        if self.logdomain:
            with np.errstate(divide = "ignore"):
                self.messages = [np.log(ErrorRates)]
        else:
            self.messages = [ErrorRates]
        self.ErrorRates = ErrorRates
        self.InitMaxSumMessages()
    def InitMessages(self,Messages = None):
        if Messages is not None:
            Node.InitMessages(self,Messages)
        else:
            if self.logdomain == True:
                with np.errstate(divide = "ignore"):
                    self.messages = [np.log(self.ErrorRates)]
            else:
                self.messages = [self.ErrorRates]
    def InitMaxSumMessages(self):
        """
        Note that if some rates are zero, the log will return -infinity for these which works fine for the max sum algorithm.
        This does however raise a numpy runtime warning: divide by zero encountered in log
        """
        with np.errstate(divide = "ignore"):
            self.maxsummessage=np.log(self.ErrorRates)
    def SetErrorRates(self,ErrorRates):
        self.ErrorRates = ErrorRates
        self.InitMessages()
        self.InitMaxSumMessages()
    def SetLogdomain(self,logdomain):
        self.logdomain = logdomain
        self.InitMessages()
        

class LogicalFactorNode(Node):
    """
    Represents a Factor in a Concatenated Quantum Code that connects the Qubits of a code instance with their logical Qubit, i.e. it contains the checks and the connection between physical and logiacal qubits
    """
    def __init__(self,BaseCode, name = "",logdomain = True):
        Node.__init__(self,name,logdomain)
        #Look up tables giving the syndrome and the logical error for each error on the child qubits. Must be in same order as in ErrorList
        #Note: These should just be references to the Tables saved in base code, dont copy the lists for each new node
        self.BaseCode = BaseCode
        self.SyndromesLUT = BaseCode.SyndromeLUT
        self.LogicalLUT = BaseCode.LogicalLUT 
        self.ErrorList = BaseCode.Errors #All possible Errors on the Child Qubits as a 2D numpy array 
   
        self.MaxSumConfiguration = [] #Will store the maximizing configuration corresponding to the maxsummessage. Used in the downwards messages in max sum algorithm
    
    def SetLogdomain(self,logdomain):
        self.logdomain = logdomain
    
    def UpdateMessageToParent(self,Syndrome):
        self.messages[-1] = self._CalculateMessageToParent(Syndrome)
    def _CalculateMessageToParent(self,Syndrome):
        """
        Calculates the outgoing message for Belief Propagation to Parent Node given the Observed Syndrome at this Factor.
        """ 
        IncomingQubitMessages = np.array([c.GetMessage(self) for c in self.children if c.NodeType == 0])
        IncomingMeasurementErrorMessages = np.array([c.GetMessage(self) for c in self.children if c.NodeType == 1])
        if len(IncomingMeasurementErrorMessages) == 0:
            IncomingMeasurementErrorMessages = None
        Syndrome = np.array(Syndrome)
        res = self._NumbaCalculateMessageToParentErrsPerSyndrome(self.BaseCode.ErrorsPerSyndrome,Syndrome,IncomingQubitMessages,self.logdomain,IncomingMeasurementErrorMessages,self.BaseCode.MeasurementErrors)
        return res
       
    @staticmethod
    @jit(nopython = True)
    def _NumbaCalculateMessageToParentErrsPerSyndrome(ErrsPerSyndrome,Syndrome,Messages,logdomain,MeasurementMessages = None,MeasurementErrorList = None):
        """
        Calculates the Belief Propagation message to parent, for a specific syndrome
        Takes the ErrsPerSyndromeList of the base code and a Syndrome, i.e. uses a lookup table to get the matching errors for each syndrome
        If measurement errors are considered,then MeasurementMessages should be the messages from the measurement error children
        Messages should only be those coming from the qubit children
        """
        OutMessage = np.zeros(4) #assumes quaternary representation
        if MeasurementMessages is None:
            """
            In this case only one syndrome is considered and we will later sum over the message products corresponding to errors matching this syndrome
            """
            ErrorList = ErrsPerSyndrome[SyndromeToIndex(Syndrome)]
            Products = _NumbaCalculateProductsForParentErrsPerSyndrome(ErrorList,Messages,logdomain)
        else:
            """
            In this case all syndromes must be considered
            """
            Products = np.empty((4,0))  #A list of all the calculated products so we can determine their max
            for i in range(MeasurementErrorList.shape[0]):
                m = MeasurementErrorList[i]
                SyndromeWithError = (m + Syndrome) % 2
                ErrorList = ErrsPerSyndrome[SyndromeToIndex(SyndromeWithError)]
                SynProducts = _NumbaCalculateProductsForParentErrsPerSyndrome(ErrorList, Messages, logdomain)
                MeasProduct = MessageProduct(MeasurementMessages,m,logdomain)
                if logdomain:
                    SynProducts += MeasProduct
                else:
                    SynProducts *= MeasProduct
                Products = np.hstack((Products,SynProducts))  
        if logdomain:
            #Renormalize messages to be less small, then transform back to normal domain
            maximum = np.max(Products)
            Products -= maximum
            Products = np.exp(Products)        
        #Do the summation over all states
        for i in range(4):
            for j in range(Products.shape[1]):
                OutMessage[i] += Products[i,j]
        if logdomain:
            OutMessage = np.log(OutMessage)
        return OutMessage

    def UpdateMessagesToChildren(self,Syndrome):
        Messages = self._CalculateMessagesToChildren(Syndrome)
        MessagesToChildren = [m for m in Messages]
        self.messages[0:len(MessagesToChildren)] = MessagesToChildren
    def _CalculateMessagesToChildren(self,Syndrome):
        """
        Calculate Belief Propagation messages for children
        """
        QubitMessagesFromChildren = np.array([c.GetMessage(self) for c in self.children if c.NodeType == 0])
        MeasurementMessagesFromChildren = np.array([c.GetMessage(self) for c in self.children if c.NodeType == 1])
        MeasurementErrorList = self.BaseCode.MeasurementErrors
        if len(MeasurementMessagesFromChildren) == 0:
            MeasurementMessagesFromChildren = None
            MeasurementErrorList = np.zeros((1,self.BaseCode.MeasurementErrors.shape[1]),dtype=int)
        MessageFromParent = np.array(self.parent.GetMessage(self))
        ErrList = self.BaseCode.ErrorsPerSyndrome
        qres,mres = self._NumbaCalculateMessagesToChildrenErrsPerSyndrome(ErrList,Syndrome,QubitMessagesFromChildren,MessageFromParent,self.logdomain,MeasurementErrorList = MeasurementErrorList,MeasurementMessagesFromChildren = MeasurementMessagesFromChildren)
        if mres is not None:
            res = list(qres) + list(mres)
        else:
            res = qres
        return res
    @staticmethod
    @jit(nopython = True)
    def _NumbaCalculateMessagesToChildrenErrsPerSyndrome(ErrsPerSyndrome,Syndrome,QubitMessagesFromChildren,MessageFromParent,logdomain, MeasurementErrorList,MeasurementMessagesFromChildren = None):
        """
        If no Measurement Errors are present then MeasurementErrorList should be the arrray [[0,0,0,0,0,...]] where the number of zeros is the number of checks, and MeasurementMessagesFromChildren should be None
        This variant uses a lookup table to get the matching errors for each syndrome
        """
        #OutMessages = np.zeros((MessagesFromChildren.shape[0],4)) #assumes quaternary representation
        QubitOutMessages = np.zeros((QubitMessagesFromChildren.shape[0],4)) #quaternary
        if MeasurementMessagesFromChildren is not None:
            MeasurementOutMessages = np.zeros((MeasurementMessagesFromChildren.shape[0],2)) #binary
        else:
            MeasurementOutMessages = None
        """
        Now we can obtain the message to child j by dividing out the message of child j from the total product. This is more efficient than summing over all messages except j for each j
        """
        ErrorList0 = ErrsPerSyndrome[0]
        n_errs = ErrorList0.shape[0]
        n_meas_errs = MeasurementErrorList.shape[0]
        QubitProducts = np.zeros((QubitMessagesFromChildren.shape[0],4,int(n_errs*n_meas_errs / 4)))
        QubitProductsIndizes = np.zeros((QubitMessagesFromChildren.shape[0],4),dtype=numba.int64)
        #QubitProductsIndizes = np.zeros((QubitMessagesFromChildren.shape[0],4),dtype=int)
        if MeasurementMessagesFromChildren is not None:
            MeasurementProducts = np.zeros((MeasurementMessagesFromChildren.shape[0],2,int(n_errs*n_meas_errs/2)))
            MeasurementProductsIndizes = np.zeros((MeasurementMessagesFromChildren.shape[0],2),dtype=numba.int64)
            #MeasurementProductsIndizes = np.zeros((MeasurementMessagesFromChildren.shape[0],2),dtype=int)
        
        if MeasurementMessagesFromChildren is not None:
            MessageLists = [QubitMessagesFromChildren,MeasurementMessagesFromChildren]
            IndexLists = [QubitProductsIndizes,MeasurementProductsIndizes]
            ProductsLists = [QubitProducts,MeasurementProducts]
        else:
            MessageLists = [QubitMessagesFromChildren]
            IndexLists = [QubitProductsIndizes]
            ProductsLists = [QubitProducts]

        for m_i in range(MeasurementErrorList.shape[0]):
            m = MeasurementErrorList[m_i]
            s = (Syndrome + m) % 2
            ErrorList = ErrsPerSyndrome[SyndromeToIndex(s)]
            measprod = MessageProduct(MeasurementMessagesFromChildren,m,logdomain) #If no messages are there this returns 1 / 0 depending on logdomain
            for i in range(ErrorList.shape[0]):
                e = ErrorList[i,:-1]
                L = ErrorList[i,-1]
                prod = MessageProduct(QubitMessagesFromChildren,e,logdomain)
                if logdomain:
                    prod += MessageFromParent[L]
                    prod += measprod                    
                else:
                    prod *= MessageFromParent[L]
                    prod *= measprod
                vectors = (e,m)
                for l in range(len(MessageLists)):
                    """
                    Decide if we iterate over the qubit / measurement children
                    """
                    MessagesFromChildren = MessageLists[l]
                    indizes = IndexLists[l]
                    Products = ProductsLists[l]
                    v = vectors[l]
                    for j in range(MessagesFromChildren.shape[0]):
                        """
                        Iterate over the qubit / measurement children and compute the product for each by dividing out the message from the total product
                        """
                        if logdomain:
                            if MessagesFromChildren[j,v[j]] != -math.inf:
                                """
                                In this case sum of messages except j'th can be computed by substracting j'th message from the sum of all messages. More efficient than summing all messages except one for every j.
                                """
                                prod_j = prod - MessagesFromChildren[j,v[j]]
                            else:
                                """
                                In this case we must manually recompute the sum of the other messages
                                """
                                mask = np.arange(0,len(MessagesFromChildren),step=1) != j
                                #MessagesFromChildren_j = np.array([MessagesFromChildren[k] for k in range(len(MessagesFromChildren)) if k != j],dtype=numba.float64) #All messages except j'th, cannot use np.delete because numba doesnt support axis argument
                                MessagesFromChildren_j = MessagesFromChildren[mask]
                                v_j = np.delete(v,[j])
                                prod_j = MessageProduct(MessagesFromChildren_j,v_j,logdomain)
                        else:
                            if MessagesFromChildren[j,v[j]] != 0: 
                                prod_j = prod / MessagesFromChildren[j,v[j]]
                            else:
                                """
                                In this case we must manually recompute the product of the other messages
                                """
                                mask = np.arange(0,len(MessagesFromChildren),step=1) != j
                                MessagesFromChildren_j = MessagesFromChildren[mask]
                                v_j = np.delete(v,[j])
                                prod_j = MessageProduct(MessagesFromChildren_j,v_j,logdomain)
                        
                        Products[j,v[j],indizes[j,v[j]]] = prod_j
                        indizes[j,v[j]] += 1
           
        if logdomain:
            #Renormalize messages to be less small, then transform back to normal domain
            for l in range(len(ProductsLists)):
                Products = ProductsLists[l]
                for i in range(Products.shape[0]): 
                    maximum = np.max(Products[i,:,:])
                    Products[i,:,:] = Products[i,:,:] - maximum #This is an inplace operation
                np.exp(Products,Products) #This MUST be done inplace since tuplelists stores a reference to a specific object, and that one must be modified. The second argument is the out= argument.
        #Compute the outmessages by summing
        for i in range(QubitMessagesFromChildren.shape[0]):
            for j in range(4):
                for k in range(QubitProducts.shape[2]):
                    QubitOutMessages[i,j] += QubitProducts[i,j,k]
        if logdomain:
            QubitOutMessages = np.log(QubitOutMessages)
        if MeasurementMessagesFromChildren is not None:
            for i in range(MeasurementMessagesFromChildren.shape[0]):
                for j in range(2):
                    for k in range(MeasurementProducts.shape[2]):
                        MeasurementOutMessages[i,j] += MeasurementProducts[i,j,k]
            if logdomain:
                if MeasurementOutMessages is not None:
                    #MeasurementOutMessages = np.log(MeasurementOutMessages)
                    for i in range(MeasurementOutMessages.shape[0]):
                        for j in range(MeasurementOutMessages.shape[1]):
                            MeasurementOutMessages[i,j] = np.log(MeasurementOutMessages[i,j])
        return QubitOutMessages,MeasurementOutMessages
        
        
    
    def UpdateMaxSumMessageToParent(self,Syndrome):
        self.maxsummessage,self.MaxSumConfiguration = self._CalculateMaxSumMessageToParent(Syndrome)
    def _CalculateMaxSumMessageToParent(self,Syndrome):
        """
        Calculates the outgoing max sum message to Parent Node given the Observed Syndrome at this Factor.
        """ 
        IncomingQubitMessages = np.array([c.GetMaxSumMessage() for c in self.children if c.NodeType == 0])
        IncomingMeasurementMessages = np.array([c.GetMaxSumMessage() for c in self.children if c.NodeType == 1])
        MeasurementErrorList = self.BaseCode.MeasurementErrors
        if len(IncomingMeasurementMessages) == 0:
            IncomingMeasurementMessages = None
            MeasurementErrorList = np.zeros((1,self.BaseCode.n_checks),dtype=int)
        res,qubitconfig,measconfig = self._NumbaMaxSumMessageToParent(self.BaseCode.ErrorsPerSyndrome,Syndrome,IncomingQubitMessages,MeasurementErrorList,IncomingMeasurementMessages)
        if IncomingMeasurementMessages is not None:
            config = np.hstack((qubitconfig,measconfig))
        else:
            config = qubitconfig
        return res,config  

    @staticmethod
    @jit(nopython=True)
    def _NumbaMaxSumMessageToParent(ErrorsPerSyndrome,Syndrome,QubitMessages,MeasurementErrorList,MeasurementMessages):
        """
        Max-Sum (also known as max-product without the log) algorithm message to parent for MAP queries. Also returns the maximizing configuration since it is needed for downwards pass
        Used to implement "Hard Assignment Method"
        If no Measurement Errors are present then MeasurementErrorList should be the arrray [[0,0,0,0,0,...]] where the number of zeros is the number of checks, and MeasurementMessages should be None
        ErrorsPerSyndrome contains a List of all errors matching a syndrome for each error, and the logical error is appended to the error in that list
        Note that we dont need to maximize over errors that do not match the syndrome since they cannot be true, i.e. the factor contributs -inf
        For the matching error the factor contributes 0 to the sum in log domain (since the factor value is 1), so its not passed to MessageSum
        """
        OutMessage = np.zeros(4) #assumes quaternary representation
        QubitOutConfiguration = np.zeros((4,ErrorsPerSyndrome.shape[2]-1),dtype=numba.int64) #Used to find the configuration corresponding to the highest probability and save it
        MeasurementOutConfiguration = np.zeros((4,MeasurementErrorList.shape[1]),dtype=numba.int64)
        Sums = np.zeros((4,int(ErrorsPerSyndrome.shape[1]*MeasurementErrorList.shape[0] / 4)))  #A list of all the calculated sums
        Configurations = np.zeros((4,int(ErrorsPerSyndrome.shape[1]*MeasurementErrorList.shape[0] / 4),3),dtype = numba.int64) #A list of all the possible configurations (as pairs of indizes in MeasurementErrorList and ErrorsPerSyndrome, where two indizes are used for errors per syndrome)
        #Shape uses the fact that there are equal amount of errs corresponding to each logop
        indizes = np.zeros((4,),dtype = numba.int64)
        #indizes = np.zeros((4,),dtype = int)
        for m_i in range(MeasurementErrorList.shape[0]):
            m = MeasurementErrorList[m_i]
            s = (Syndrome + m)  % 2
            syndromeindex = SyndromeToIndex(s)
            ErrorList = ErrorsPerSyndrome[syndromeindex]
            MeasurementSum = MessageSum(MeasurementMessages,m) #Returns 0 if the messages were None
            for i in range(ErrorList.shape[0]):
                e = ErrorList[i,:-1]
                L = ErrorList[i,-1]
                QubitSum = MessageSum(QubitMessages,e)
                Sum = QubitSum + MeasurementSum
                Sums[L,indizes[L]] = Sum
                Configurations[L,indizes[L],0] = m_i
                Configurations[L,indizes[L],1] = syndromeindex
                Configurations[L,indizes[L],2] = i
                indizes[L] += 1
        for L in range(4):
            x = np.argmax(Sums[L,:])
            OutMessage[L] = Sums[L,x]
            QubitOutConfiguration[L] = ErrorsPerSyndrome[Configurations[L,x,1],Configurations[L,x,2],:-1] #Last entry is excluded because its the logical error, not the physical
            MeasurementOutConfiguration[L] = MeasurementErrorList[Configurations[L,x,0],:]
        return OutMessage,QubitOutConfiguration,MeasurementOutConfiguration  
    def UpdateChildrenMaxSum(self):
        x = self.parent.state 
        MAPConfig = self.MaxSumConfiguration[x]
        for i,n in enumerate(self.children):
            n.state = MAPConfig[i]
            
    def PropagateError(self):
        """
        Used for Syndrome Computation,not Belief Propagation.
        Does not inherently take into account measurement errors
        Updates the state of the parent variable based on the states of the qubit child variables by computing the logical error.
        Returns the syndrome corresponding to the state of the childs
        """
        Syndrome,LogicalError = self._GetSyndromAndLogicalError()
        self.parent.state = LogicalError
        return Syndrome
    def _GetQubitChildStates(self):
        return np.array([c.state for c in self.children if c.NodeType == 0])
    def _GetSyndromAndLogicalError(self):
        state = self._GetQubitChildStates()
        Syndrome,LogicalError = self.BaseCode.GetSyndromeAndLogicalError(state)
        return Syndrome,LogicalError

@jit(nopython=True)
def _NumbaCalculateProductsForParentErrsPerSyndrome(ErrorList,Messages,logdomain):
    """
    Calculates the message products corresponding to all the configurations of qubit states for each possible logical state of the parent needed for the message (message is calculated by summing over these) 
    Used in LogicalFactorNode
    """ 
    Products = np.zeros((4,int(ErrorList.shape[0] / 4)))  #A list of all the calculated products so we can determine their max
    #Shape uses the fact that there are equal amount of errs corresponding to each logop
    indizes = np.zeros((4,),dtype = numba.int64)
    for i in range(ErrorList.shape[0]):
        e = ErrorList[i,:-1]
        L = ErrorList[i,-1]
        prod = MessageProduct(Messages,e,logdomain)
        if math.isnan(prod):
            print("Messages: ", print(Messages))
            raise ValueError("Encountered Nan")
        Products[L,indizes[L]] = prod
        indizes[L] += 1
    return Products
@staticmethod
@jit(nopython = True)
def _NumbaCalculateTotalMessageProductQubitChildrenErrsPerSyndrome(ErrorList,QubitMessagesFromChildren,MessageFromParent,logdomain):
    """
    Caclulcates the message product corresponding to all configuration of qubit states for a given syndrome / measurement error needed for children messages (children message is calculated by dividing one of the messages out and then summing)
    This variant uses a lookup table to get the matching errors for each syndrome
    Used In LogicalFactorNode
    """
    Products = np.zeros(int(ErrorList.shape[0] / 4))  #A list of all the calculated products so we can determine their min / max
    #Shape uses the fact that there are equal amount of errs corresponding to each logop
    #indizes = np.zeros((QubitMessagesFromChildren.shape[0],4),dtype = numba.int64)
    for i in range(ErrorList.shape[0]):
        e = ErrorList[i,:-1]
        L = ErrorList[i,-1]
        prod = MessageProduct(QubitMessagesFromChildren,e,logdomain)
        if logdomain:
            prod += MessageFromParent[L]
        else:
            prod *= MessageFromParent[L]
        Products[i] = prod
    return Products
        
@jit(nopython = True)
def MessageProduct(Messages,e,logdomain = True):
    """
    Used for Belief Propagation (Sum product)
    """
    if logdomain:
        prod = 0
    else:
        prod = 1
    if Messages is not None:
        for j in range(Messages.shape[0]):
            if logdomain:
                prod += Messages[j,e[j]]
            else:
                prod *= Messages[j,e[j]]
    return prod

@jit(nopython = True)
def MessageSum(Messages,e):
    """
    Used for max sum algorithm, Messages should be in log domain
    """
    Sum = 0
    if Messages is not None:
        for j in range(Messages.shape[0]):
            Sum += Messages[j,e[j]]
    return Sum

@jit(nopython = True)
def SyndromesEqual(syn1,syn2):
    for i in range(syn1.shape[0]):
        if syn1[i] != syn2[i]:
            return False
    return True
