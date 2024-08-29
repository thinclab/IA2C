'''====================================================================================
Belief Filter for IA2C. Will propagate model beliefs for one step given a batch of
observations from vectorized environments. Uses randomly generated models in the
constructor, but also includes option to use known models for Org & HVT domains.

Copyright (C) March, 2024  Bikramjit Banerjee

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

===================================================================================='''

import numpy as np

def generate_random_probability_matrix(m, n):
    matrix = np.random.rand(m, n)
    matrix /= np.sum(matrix, axis=1)[:, np.newaxis]
    return matrix

class BeliefFilter():
    def __init__(self, num_models, num_actions, num_envs):
        self.prior = np.tile(np.ones(num_models)*round(1.0/num_models, 2), (num_envs, 1))

        #============ORG Domain (used in paper's expts) ==========================================
        #self.filters = np.array([[0.8,0.6,0.4],[0.1,0.2,0.3],[0.1,0.2,0.3]])
        #self.filterAction = np.array([[0.8,0.1,0.1],[0.6,0.2,0.2],[0.4,0.3,0.3]])

        #============HVT Domain===================================================================
        #self.filters = np.array([[0.8,0.6,0.4,0.2,0.1], [0.05,0.1,0.15,0.2,0.225], [0.05,0.1,0.15,0.2,0.225], [0.05,0.1,0.15,0.2,0.225], [0.05,0.1,0.15,0.2,0.225]])
        #self.filterAction = np.array([[0.8,0.05,0.05,0.05,0.05],[0.6,0.1,0.1,0.1,0.1],[0.4,0.15,0.15,0.15,0.15],[0.2,0.2,0.2,0.2,0.2],[0.1,0.225,0.225,0.225,0.225]])

        self.filterAction = generate_random_probability_matrix(num_models, num_actions) #Use above if known models are desired. This will generate random models.
        self.filters = self.filterAction.transpose() #(N_A X N_M)
        
        self.num_actions = num_actions
        self.num_models = num_models
    
    def update(self, obs, prev_belief): #(N_E X N_A), (N_E X N_M)
        #print("----------", obs.shape, prev_belief.shape)
        batch_size = np.shape(obs)[0]
        I = np.tile(np.eye(self.num_models), (batch_size, 1))
        I = np.reshape(I, (batch_size, self.num_models, self.num_models)) #(N_E X N_M X N_M)
        belief = I * np.reshape(prev_belief, (batch_size, self.num_models, 1)) #(N_E X N_M X N_M)
        result = np.matmul(self.filters, belief) #(N_E X N_A X N_M)
        bp = np.einsum('ij,ijk->ik', obs, result)
        bprime = bp/np.sum(bp, axis=-1, keepdims=True)
        prediction = np.matmul(bprime, self.filterAction)

        c = prediction.cumsum(axis=-1)
        u = np.random.rand(len(c), 1)
        ap = (u < c).argmax(axis=-1)
        bprime = (bp/np.sum(bp, axis=-1, keepdims=True)).round(2)
        return ap, bprime, prediction
