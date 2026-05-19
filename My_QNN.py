from mindquantum.core.gates import *
from mindquantum.core.circuit import *
from mindquantum.simulator import Simulator
from mindquantum.core import Measure
from mindspore import Tensor

import numpy as np
from numpy import pi,cos,sin
import time

import copy

"""1. Define the QNN class"""
class QNN():
    def __init__(self, QCircuit, params_name_dict, shots=1024, measure_type="state_vector", 
                 optimizer=None):
        self.params_name=params_name_dict["params_name"] # All parameters
        self.encode_params_name=params_name_dict["encode_params_name"] # Parameters used for encoding
        self.weight_params_name=params_name_dict["weight_params_name"] # Parameters used for computation
        self.valid_params_name=self.encode_params_name.copy()+self.weight_params_name.copy()
        
        self.params={key:0 for key in self.params_name}
        self.init_params=self.params.copy()
        
        self.shots=shots
        self.results=dict()
        self.measure_type=measure_type
        
        self.qc= QCircuit
        self.nq = self.qc.n_qubits
        self.n_vq = 0
        self.sim = Simulator('mqvector', self.qc.n_qubits) 
        
        self.grad=None
        self.grad_combined=None
        
        self.optimizer=optimizer
        
        
    def initialize_parameters(self,params=None,random_seed=1,random_type="normal",amp = 0.1):
        np.random.seed(random_seed)
        """Function: Initialize the parameters of the circuit"""
        if params is None:
            for key in self.valid_params_name:
                if random_type=="normal":
                    self.init_params[key]=np.random.normal(loc=0,scale=0.5)  
                elif random_type=="uniform":
                    self.init_params[key]=np.random.rand()*np.pi*amp 
        else:
            self.init_params=params.copy
        self.params=self.init_params.copy()
        return self.init_params
    
    def forward(self,input_data=None,params=None,shots=None):
        """ Forward propagation: data dimension (batch_size, 2**n_qubits)"""
        if input_data is None:
            input_data=np.zeros((1,len(self.encode_params_name))).tolist()
        if params is None:
            params=self.params.copy()
        if shots is None:
            shots=self.shots

        Probs=[]
        for n in range(len(input_data)):
            params_ls=params.copy()
            for idx,name in enumerate(self.encode_params_name):
                if name in self.weight_params_name:
                    params_ls[name] += input_data[n][idx]
                else:
                    params_ls[name]=input_data[n][idx]

            P_dict=params_ls.copy()

            """Run"""

            if self.measure_type=="sampling":
                self.sim.reset() # Reset the Simulator
                sample_data = self.sim.sampling(self.qc, P_dict, shots=shots).data
                self.results = copy.deepcopy(sample_data)

                """# Process the results"""
                results = []
                for k in range(0,2**(self.nq)):
                    key=bin(k)[2:].zfill(self.nq)
                    
                    if key not in sample_data.keys():
                        results.append(0)
                    else:
                        results.append(results[key])
                       
                results = np.array(results)
                results = results[0:2**(self.nq - self.n_vq)]

            elif self.measure_type=="state_vector":

                self.sim.reset() # Reset the Simulator
                self.sim.apply_circuit(self.qc, pr=P_dict)
                results=self.sim.get_qs(False)
                results=abs(results)**2
                results=results[0:2**(self.nq - self.n_vq)]
                
            results += 1e-9
            results=results/sum(results)
            self.results=copy.deepcopy(results)
            Probs.append(results)

        Probs=np.array(Probs)

        return Probs
    
    
    def backward(self,input_data=None,params=None,shots=None,target_data = None,
                 cost_fn = None,ancella_data = [6,7]):
        """Backward propagation: data dimension (batch_size, output_shape, num_weights)"""
        if input_data is None:
            input_data=np.zeros((1,len(self.encode_params_name))).tolist()
        if params is None:
            params=self.params.copy()
        if shots is None:
            shots=self.shots
            
        optim = self.optimizer.optim
        
        num_weights=len(self.weight_params_name)
        
        grads=[]
        
        if optim in ["Adam","RMSprop","AMSGrad"]:
            # defalut "parameter-shift rule" for grad evaluation
            for weight_name in self.weight_params_name:
                plus_params=copy.deepcopy(params)
                minus_params=copy.deepcopy(params)

                plus_params[weight_name]+=np.pi/2
                minus_params[weight_name]-=np.pi/2
                eps = 1

                plus_dist=self.forward(input_data,params=plus_params,shots=shots)
                minus_dist=self.forward(input_data,params=minus_params,shots=shots)

                if cost_fn is None:
                    grad=(plus_dist-minus_dist)/(2*eps)
                else:
                    grad = (cost_fn(plus_dist, target_data,ancella_data) - cost_fn(minus_dist, target_data,ancella_data))/(2*eps)
                grads.append(grad)
                
                return np.array(grads).reshape(-1,)
                
        if optim in ["SPSA","SPSA-Adam","SPSA-RMSprop","SPSA-AMSGrad"]:
            # defalut "SASA" for grad evaluation
            
            n_weights = len(self.weight_params_name)
            n_spsa_sample = self.optimizer.n_spsa_sample # Sample multiple times to improve the accuracy of gradient estimation
            
            ak = self.optimizer.ak
            ck = self.optimizer.ck
                
            grads_ave = []
            
            for k in range(n_spsa_sample):
                # Generate a random perturbation vector (values ±1)
                delta = 2 * np.random.randint(0, 2, n_weights) - 1

                # Compute the loss function on both sides
                plus_params=copy.deepcopy(params)
                minus_params=copy.deepcopy(params)

                for i,weight_name in enumerate(self.weight_params_name):
                    #print(plus_params[weight_name],delta[i])
                    plus_params[weight_name] += ck * delta[i]
                    minus_params[weight_name] -= ck * delta[i]

                plus_dist = self.forward(input_data,params=plus_params,shots=shots)
                minus_dist = self.forward(input_data,params=minus_params,shots=shots)
                
                grads = (cost_fn(plus_dist, target_data, ancella_data) - cost_fn(minus_dist, target_data, ancella_data))/(2*ck*delta)
                grads = np.array(grads)
                grads_ave.append(grads)
                
            grad_ave = np.mean(np.array(grads_ave),axis = 0)
 
            return grad_ave
            
    def step(self,grad_combined=None,optim=None):
        """Update parameters once"""
        if grad_combined is None:
            grad_combined=self.grad_combined
        if optim is None:
            optim=self.optimizer.optim
        
        grad_combined=np.real(grad_combined)
        
        if optim=="Adam":
            grad_=self.optimizer.Adam(grad_combined).copy()
        elif optim=="RMSprop":
            grad_=self.optimizer.RMSprop(grad_combined).copy()
        elif optim=="AMSGrad":
            grad_=self.optimizer.AMSGrad(grad_combined).copy()
        elif optim=="SPSA":
            grad_=self.optimizer.SPSA(grad_combined).copy()
  
        elif optim == "SPSA-Adam":
            grad_=self.optimizer.SPSA(grad_combined).copy()
            grad_=self.optimizer.Adam(grad_).copy()
        elif optim == "SPSA-RMSprop":
            grad_=self.optimizer.SPSA(grad_combined).copy()
            grad_=self.optimizer.RMSprop(grad_).copy()
        elif optim == "SPSA-AMSGrad":
            grad_=self.optimizer.SPSA(grad_combined).copy()
            grad_=self.optimizer.AMSGrad(grad_).copy()
            
        for idx,weight_name in enumerate(self.weight_params_name):
            self.params[weight_name]-=grad_[idx]
        return self.params
    
    
"""2. Define the optimizer class"""
class Optimizer():
    def __init__(self, learning_rate=None,beta=None,epsilon=None, optim="Adam", spsa_params = {"c":0.2, "min_ck": 0.1, "A":30, "n_spsa_sample":1},beta_spsa=None):
        
        self.m_t=None
        self.v_t=None
        self.learning_rate=learning_rate
        self.beta=beta
        self.epsilon=epsilon
        self.optim=optim
        
        self.c = spsa_params["c"]
        self.ak = self.learning_rate
        self.A = spsa_params["A"] # SPSA stabilization coefficient
        self.ck = self.c
        self.min_ck = spsa_params["min_ck"]
        self.n_spsa_sample = spsa_params["n_spsa_sample"]
        self.beta_spsa = beta_spsa
        
        self.t = 0 # Optimization step
        
    def RMSprop(self,grad,learning_rate=0.1,beta=0.99,epsilon=1e-10):
        """
        Implement the RMSProp optimizer

        Parameters:
            grad: gradient function, receives the current gradient.
            learning_rate: learning rate.
            beta: momentum parameter.
            epsilon: numerical stability constant, usually very small.

        Returns:
            grad_: optimized gradient.
        """
        if self.learning_rate!=None:
            learning_rate=self.learning_rate
        if self.beta is not None:
            beta=self.beta
        if self.epsilon is not None:
            epsilon=self.epsilon
        
        if self.v_t is None:
            v_t=np.zeros_like(grad)
        else:
            v_t = self.v_t.copy()
                
        v_t=beta*v_t+(1-beta)*(grad**2)
        v_t_hat = v_t / (1 - beta)
        grad_=learning_rate*grad/(np.sqrt(v_t_hat)+epsilon)
        
        self.v_t=v_t.copy()
        return grad_
    
    
    def Adam(self,grad=None,learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Implement the Adam optimizer

        Parameters:
            grad: gradient function, receives the current gradient.
            learning_rate: learning rate.
            beta1: momentum parameter.
            beta2: second-moment parameter.
            epsilon: numerical stability constant, usually very small.

        Returns:
            grad_: optimized gradient.
        """
        if self.learning_rate!=None:
            learning_rate=self.learning_rate
        if self.beta is not None:
            beta1=self.beta[0]
            beta2=self.beta[1]
        if self.epsilon is not None:
            epsilon=self.epsilon
            
        if self.m_t is None:
            m_t=np.zeros_like(grad)
            v_t=np.zeros_like(grad)
        else:
            m_t = self.m_t.copy()
            v_t = self.v_t.copy()
        

        m_t=beta1*m_t+(1 - beta1) * grad
        v_t=beta2*v_t+(1-beta2)*(grad**2)

        m_t_hat = m_t / (1 - beta1)
        v_t_hat = v_t / (1 - beta2)

        grad_=learning_rate*m_t_hat/(np.sqrt(v_t_hat)+epsilon)

        self.m_t=m_t.copy()
        self.v_t=v_t.copy()

        return grad_
    
    
    def AMSGrad(self,grad=None,learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Implement the Adam optimizer

        Parameters:
            grad: gradient function, takes the current gradient.
            learning_rate: learning rate.
            beta1: momentum parameter.
            beta2: second-moment parameter.
            epsilon: numerical stability constant, usually very small.

        Returns:
            grad_: optimized gradient.
        """
        if self.learning_rate!=None:
            learning_rate=self.learning_rate
        if self.beta is not None:
            beta1=self.beta[0]
            beta2=self.beta[1]
        if self.epsilon is not None:
            epsilon=self.epsilon
            
        if self.m_t is None:
            m_t=np.zeros_like(grad)
            v_t=np.zeros_like(grad)
            v_hat_max = np.zeros_like(grad)
        else:
            m_t = self.m_t.copy()
            v_t = self.v_t.copy()
            v_hat_max = self.v_hat_max
        

        m_t=beta1*m_t+(1 - beta1) * grad
        v_t=beta2*v_t+(1-beta2)*(grad**2)

        m_t_hat = m_t / (1 - beta1)
        v_t_hat = v_t / (1 - beta2)
        
        v_hat_max = np.maximum(v_hat_max, v_t_hat)

        grad_=learning_rate*m_t_hat/(np.sqrt(v_hat_max)+epsilon)

        self.m_t=m_t.copy()
        self.v_t=v_t.copy()
        self.v_hat_max = v_hat_max.copy()

        return grad_
    
    
    def SPSA(self, grad=None,learning_rate=0.1, c=0.1, alpha=0.602, gamma=0.101):
        if self.learning_rate!=None:
            learning_rate=self.learning_rate
        if self.beta_spsa is not None:
            alpha=self.beta_spsa[0] # learning rate decay coefficient
            gamma=self.beta_spsa[1] # Perturbation decay coefficient
            
        if self.c is not None:
            c = self.c
        
        ak = learning_rate/(self.t + 1+self.A)**alpha
        ck = c/(self.t + 1)**gamma
        
        self.ak = ak
        self.ck = max(ck,self.min_ck)
        self.t += 1
        
        grad_ = copy.deepcopy(grad)

        return grad_
   