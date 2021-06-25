import numpy as np
import math
from utils import maxentsoftmax, Struct
import numpy.random as random
from itertools import product


class MDP(object):

    def __init__(
    self, 
    game, 
    ):

        self.game = game

    def linearvalueiteration(self,r):

        states = self.game.states
        actions = self.game.actions
        discount = self.game.discount
        sa_p = self.game.sa_p
        sa_s = self.game.sa_s
        VITR_THRESH = 1e-4

        v = np.zeros((states,1))

        diff = 1
        while diff >= VITR_THRESH:

            vp = v
            temp1 = np.squeeze(vp[sa_s], axis=3)
            temp2 = np.multiply(sa_p, temp1)
            q = r + discount*np.sum(temp2, axis=2)
            

            v = maxentsoftmax(q)
            
            diff = max(abs(v - vp))

        logp = q - np.tile(v,(1,actions))
        p = np.exp(logp)
        

        return v, q, p, logp

    def linearmdpsolve(self, r):
        v, q, p, _ = self.linearvalueiteration(r)

        mdp_solution = Struct()
        mdp_solution.v = v
        mdp_solution.q = q
        mdp_solution.p = p
        return mdp_solution


    def sampleexamples(self, mdp_solution, training_samples, training_sample_length = 8):

        random.seed(0)

        states = self.game.states
        N = training_samples
        T = training_sample_length
        example_samples = [ [] for i in range(N)]

        for i in range(N):

            s = math.ceil(random.rand(1,1)*states) - 1

            for t in range(T):

                a = self.linearmdpaction(mdp_solution,s)

                example_samples[i].append([s,a]) 

                s = self.linearmdpstep(s,a)

        return example_samples

    def sampleprediction(self, solution, length, start_s):

        prediction_sample = []
        s = start_s

        for t in range(length+1):

            a = self.linearmdpaction(solution,s)

            prediction_sample.append([s,a]) 

            s = self.linearmdpstep(s,a)


        return prediction_sample

    def linearmdpaction(self, mdp_soloution, s):

        actions = self.game.actions
        samp = random.rand(1,1)
        total = 0

        for a in range(actions):

            total += mdp_soloution.p[s,a]
            if total >= samp:

                return a

    def linearmdpstep(self, s, a):

        sa_p = self.game.sa_p
        sa_s = self.game.sa_s
        r = random.rand(1,1)
        sm = 0
        temp = np.shape(sa_p)[2]

        for k in range(temp):
            sm = sm + sa_p[s,a,k]

            if sm >= r:
                s = sa_s[s,a,k]
                return s

    def linearmdpfrequency(self, mdp_solution, initD):

        sa_s = self.game.sa_s
        sa_p = self.game.sa_p
        discount = self.game.discount
        temp = np.sum(initD)

        states, actions, transitions = np.shape(sa_p)

        VITR_THRESH = 1e-4

        D = np.zeros((states,1))

        diff = 1
        temp7 = np.sum(initD)
        while diff >= VITR_THRESH:

            Dp = D
            temp1 = np.tile(np.expand_dims(mdp_solution.p, axis=2),(1,1,transitions))
            temp2 = temp1*sa_p
            temp3 = np.tile(np.expand_dims(Dp, axis=2), (1, actions, transitions))
            temp4 = temp3*discount
            DPi = temp2*temp4

            temp1 = np.reshape(sa_s, (states*actions*transitions,1))
            temp2 = np.expand_dims(np.arange(states*actions*transitions), axis=1)
            temp3 = np.reshape(DPi, (states*actions*transitions,1))
            temp4 = np.zeros((states, states*actions*transitions))
            temp4[temp1[:,0],temp2[:,0]] = temp3[:,0]
            temp5 = np.sum(np.matmul(temp4,np.ones((states*actions*transitions,1))),axis=1)
            temp6 = np.sum(temp5)
            
            D = initD + np.expand_dims(temp5, axis=1)
            

            diff = max(abs(D[:,0] - Dp[:,0]))
        temp7 = np.sum(D)

        return D


    def stdmdpfrequency(self, mdp_solution, initD):

        sa_p = self.game.sa_p
        sa_s = self.game.sa_s
        discount = self.game.discount
        temp = np.sum(initD)
        T = 8
        

        states, actions, transitions = np.shape(sa_p)


        p = mdp_solution.p
        p_value = np.argmax(p, axis=1)
        p_index = np.arange(states)
        policy = np.zeros((states,actions), int)
        policy[p_index, p_value] = 1 


        expected_svf = np.tile(initD, (1, T))

        for t in range(1, T):
            expected_svf[:, t] = 0
            for i, j, k in product(range(states), range(actions), range(transitions)):
                temp1 = expected_svf[i, t-1]
                temp2 = policy[i, j]
                temp3 = sa_p[i, j, k]
                s = sa_s[i, j, k]
                expected_svf[s, t] += temp1 * temp2 * temp3 
        expected_svf = np.expand_dims(expected_svf.sum(axis=1),axis=1)
        tempS = np.sum(expected_svf)

        return expected_svf

    def stdvalueiteration(self,r):

        states = self.game.states
        actions = self.game.actions
        discount = self.game.discount
        sa_p = self.game.sa_p
        sa_s = self.game.sa_s
        VITR_THRESH = 1e-4

        vn = np.zeros((states,1))

        diff = 1
        while diff >= VITR_THRESH:

            vp = vn
            temp1 = np.squeeze(vp[sa_s], axis=3)
            temp2 = np.multiply(sa_p, temp1)
            q = r + discount*np.sum(temp2, axis=2)
            vn = np.expand_dims(np.max(q,axis=1), axis=1)
                        
            diff = max(abs(vn - vp))

        v = vn
        temp1 = np.squeeze(vp[sa_s], axis=3)
        temp2 = np.multiply(sa_p, temp1)
        q = r + discount*np.sum(temp2, axis=2)
        p_value = np.argmax(q, axis=1)
        p_index = np.arange(states)
        p = np.zeros((states,actions), int)
        p[p_index, p_value] = 1 
        logp = []
        

        return v, q, p, logp

    def stdmdpsolve(self, r):
        v, q, p, _ = self.stdvalueiteration(r)

        mdp_solution = Struct()
        mdp_solution.v = v
        mdp_solution.q = q
        mdp_solution.p = p
        return mdp_solution

    def stdvalue_from_policy(self,r,policy):

        states = self.game.states
        actions = self.game.actions
        discount = self.game.discount
        sa_p = self.game.sa_p
        sa_s = self.game.sa_s
        VITR_THRESH = 1e-4

        vn = np.zeros((states,1))

        diff = 1
        while diff >= VITR_THRESH:

            vp = vn
            temp1 = np.squeeze(vp[sa_s], axis=3)
            temp2 = np.multiply(sa_p, temp1)
            temp3 = np.sum(temp2, axis=2)
            temp4 = r + discount*temp3
            q = np.multiply(policy, temp4)
            vn = np.expand_dims(np.sum(q,axis=1), axis=1)
                        
            diff = max(abs(vn - vp))

        v = np.mean(vn)

        
        return v
