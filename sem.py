import numpy as np
import numpy.random as random
from utils import Struct, ToTensor
from rewardmodel import BaseRewardModel,DynamicRewardModel
import torch
import torch.optim as optim
from torch.distributions.categorical import Categorical

class SEM(object):

    def __init__(
    self, 
    game,
    true_rewards,
    model, 
    linmodel_solutions,
    example_samples,
    n_samples, 
    clusters,
    alpha,
    lr=0.001,
    ):

        self.game = game
        self.true_rewards = true_rewards
        self.model = model
        self.mdpmodel_solutions = linmodel_solutions
        self.example_samples = example_samples
        self.n_samples = n_samples
        self.clusters = clusters
        self.alpha = alpha
        self.lr = lr
        self.mu_s, self.mu_sa, self.initD, self.F, self.P_k_0 = self.initial_SVF(clusters)
        
    def momaxentrun(self, maxIter = 500):

        K = self.clusters
        M, T = np.shape(self.example_samples)[0:2]
        actions = self.game.actions
        states = self.game.states
        features = np.shape(self.F)[1]
        baserewardmodel = BaseRewardModel(features)
        baserewardmodel.train()

        baseoptimizer = optim.Adam(baserewardmodel.parameters(), lr=0.001)
        dynets = []
        dyoptims = []
        for k in range(K):
            dynet = DynamicRewardModel()
            dynet.train()
            dynets.append(dynet)
            dyoptims.append(optim.Adam(dynet.parameters(), lr=self.lr)) 

        


        P_k_true = []
        for i in range(len(self.true_rewards)):
            P = np.zeros((self.n_samples[0],len(self.true_rewards)))
            P[:,i] = 1
            P_k_true.append(P)
        P_k_true_reward = np.concatenate(P_k_true, axis=0)
        P_k_true_policy = np.copy(P_k_true_reward)
        EV_n_true,_ = self.Espected_value_per_n(self.mdpmodel_solutions, P_k_true_reward, P_k_true_policy)

        min_EVD = np.inf
        min_r = 0
        min_r_seq = np.arange(K)
        

        for i in range(maxIter):


            
            reward_f = baserewardmodel(ToTensor(self.F))
            policies = []
            logpolicies = []
            rewards = []
            D_k = []
            solutions = []
            

            for k in range(self.clusters):
                dynet = dynets[k]
                r_k = dynet(reward_f)
                _, _, policy, logpolicy = self.model.linearvalueiteration(r_k.detach().numpy())
                policies.append(policy)
                logpolicies.append(logpolicy)
                rewards.append(r_k)

            
            M_k, mu_s_k, mu_sa_k, initD_k,P_k, rewards, policies, logpolicies, dynets, dyoptims = self.posterior_cluster(
                rewards, policies, logpolicies, self.P_k_0, dynets, dyoptims, reward_f
            )
            self.clusters = len(policies)
            self.P_k_0 = P_k
            st_logpolicies = np.stack(logpolicies,axis=2)
            val = np.sum(np.expand_dims(np.sum(st_logpolicies*mu_sa_k, axis=(0,1)), axis=0)/(M*T))

            baseoptimizer.zero_grad()
            for k in range(self.clusters):
                curr_optim = dyoptims[k]
                curr_optim.zero_grad()
                solution = Struct()
                solution.p = policies[k]

                curr_initD = np.expand_dims(initD_k[:,k], axis=1)
                curr_D = self.model.linearmdpfrequency(solution, curr_initD)
                D_k.append(curr_D)
                solutions.append(solution)

            D_k = np.concatenate(D_k, axis=1)
            rewards = torch.cat(rewards, axis=1)
            flat_rewards = rewards.view(-1,1)
            grad = -ToTensor(mu_s_k - D_k).view(-1,1)
            flat_rewards.backward(grad)
            for k in range(self.clusters):
                curr_optim = dyoptims[k]
                curr_optim.step()

            baseoptimizer.step()

            EV_n_pred, r_seq = self.Espected_value_per_n(solutions, P_k_true_reward, P_k)
            EVD = np.mean(abs(EV_n_true - EV_n_pred))
            Loss = -val
            Est_phi = M_k/M
                
            if EVD < min_EVD:
                min_EVD = EVD
                min_r = rewards.detach().numpy()
                min_r_seq = r_seq
                min_baserewardmodel = baserewardmodel
                min_dynets = list.copy(dynets)
                min_clusters = self.clusters
                min_Est_phi = Est_phi
            if (i+1) % 1 == 0:
                print('{:03d}/{:03d}    |   loss = {:.5f}    |   EVD = {:.5f}    |   min_EVD = {:.5f}   |   num of rewards = {:02d}'.format(
                    i+1,maxIter,Loss.item(),EVD.item(),min_EVD, len(Est_phi)
                ))
                print('demonstration shares: \n',Est_phi)


        irl_solutions = []
        R = min_r
        for k in range(min_clusters):
            r_k = np.expand_dims(R[:,k], axis=1)
            r = np.tile(r_k, (1, actions))
            soln = self.model.linearmdpsolve(r)
            v = soln.v
            q = soln.q
            p = soln.p

            irl_solution = Struct()
            irl_solution.r = r
            irl_solution.v = v
            irl_solution.q = q
            irl_solution.p = p
            irl_solution.baserewardmodel = min_baserewardmodel
            irl_solution.dynet = min_dynets[k]
            irl_solution.min_Est_phi = min_Est_phi
            irl_solutions.append(irl_solution)
            

        return irl_solutions, min_EVD, min_r_seq


    def initial_SVF(self,clusters):

        sa_s = self.game.sa_s
        sa_p = self.game.sa_p
        discount = self.game.discount
        states, actions, transitions = np.shape(sa_p)
        F = self.game.feature_data.splittable
        F = np.concatenate((F, np.ones((states,1))), axis=1)
        N = len(self.example_samples)
        T = len(self.example_samples[0])
        mu_s = np.zeros((states,N))
        mu_sa = np.zeros((states, actions, N))
        for n in range(N):
            for t in range(T):
                s = self.example_samples[n][t][0]
                a = self.example_samples[n][t][1]
                mu_s[s,n] += 1
                mu_sa[s,a,n] += 1
        initD = np.copy(mu_s)
        for n in range(N):
            for t in range(T):
                s = self.example_samples[n][t][0]
                a = self.example_samples[n][t][1]
                for k in range(transitions):
                    sp = sa_s[s,a,k]
                    ap = sa_p[s,a,k]
                    initD[sp, n] += -discount*ap

        K = self.clusters       
        P_k_0 = np.zeros((N,K))
        for n in range (N):
            r = random.randint(K)
            P_k_0[n,r] = 1
        return mu_s, mu_sa, initD, F, P_k_0


    def posterior_cluster(
        self, rewards, policies, logpolicies, P_k_0, dynets, dyoptims, reward_f
    ):
        
        N = len(self.example_samples)
        T = len(self.example_samples[0])
        states = self.game.states
        actions = self.game.actions
        
        for n in range(N):
            prior_n, rewards, policies, logpolicies, dynets, dyoptims = self.prior_cluster(
                P_k_0, rewards, policies, logpolicies, dynets,dyoptims,n, reward_f
            )
            K = len(policies)
            P_tau_n = np.ones((1,K))
            st_policies = np.stack(policies,axis=2)
            for k in range(K):
                if prior_n[k] == 0:
                    P_tau_n[0,k] = 0
                    continue 
                for t in range(T):
                    s = self.example_samples[n][t][0]
                    a = self.example_samples[n][t][1]
                    P_tau_n[0,k] *= st_policies[s, a, k]

            P_k_n = np.transpose(prior_n)*P_tau_n
            sum_k = np.expand_dims(np.sum(P_k_n, axis=1), axis=1)
            sum_k = np.tile(sum_k, (1,K))
            P_k_n = P_k_n/sum_k
            catdis = Categorical(torch.from_numpy(P_k_n))
            eta_n = catdis.sample()
            P_k_0, rewards, policies, logpolicies, dynets, dyoptims = self.update_P_k(
                P_k_0, np.array(eta_n), n, rewards, policies, logpolicies, dynets, dyoptims
            )
        K = len(policies)
        P_k = P_k_0
        M_k = np.expand_dims(np.sum(P_k, axis=0), axis=1)

        temp1 = np.tile(np.expand_dims(self.mu_s, axis=2), (1,1,K))
        temp2 = np.tile(np.expand_dims(P_k, axis=0), (states,1,1))
        mu_s_k = temp1*temp2
        mu_s_k = np.sum(mu_s_k, axis=1)

        temp1 = np.tile(np.expand_dims(self.initD, axis=2), (1,1,K))
        temp2 = np.tile(np.expand_dims(P_k, axis=0), (states,1,1))
        initD_k = temp1*temp2
        initD_k = np.sum(initD_k, axis=1)


        temp1 = np.tile(np.expand_dims(self.mu_sa, axis=3), (1,1,1,K))
        temp2 = np.tile(P_k[np.newaxis, np.newaxis, :, :], (states,actions,1,1))
        mu_sa_k = temp1*temp2
        mu_sa_k = np.sum(mu_sa_k, axis=2)


        return M_k, mu_s_k, mu_sa_k, initD_k, P_k, rewards, policies, logpolicies, dynets, dyoptims


    def prior_cluster(
        self, P_k, rewards, policies, logpolicies, dynets,dyoptims,n, reward_f
    ):
        curr_P_k = np.copy(P_k)
        P_k_minus_n = np.delete(curr_P_k,n,0)
        M_k_minus_n = np.expand_dims(np.sum(P_k_minus_n, axis=0), axis=1)

        M = len(self.example_samples)
        alpha = self.alpha
        if len(dynets) >= 3*len(self.true_rewards):
            alpha = 0
        curr_priors = M_k_minus_n/(M - 1 + alpha)
        new_prior = np.expand_dims(np.array([alpha/(M - 1 + alpha)]),axis=1)
        prior_n = np.concatenate( (curr_priors, new_prior), axis=0 )
        dynet = DynamicRewardModel()
        dynet.train()
        dyoptim = optim.Adam(dynet.parameters(), lr=self.lr)
        r_new_k = dynet(reward_f)
        _, _, policy, logpolicy = self.model.linearvalueiteration(r_new_k.detach().numpy())
        rewards.append(r_new_k)
        policies.append(policy)
        logpolicies.append(logpolicy)
        dynets.append(dynet)
        dyoptims.append(dyoptim) 
        return prior_n, rewards, policies, logpolicies, dynets, dyoptims

    def update_P_k(self, P_k, eta_n, n, rewards, policies,logpolicies,dynets,dyoptims):

        M,curr_k = np.shape(P_k)
        if eta_n > curr_k - 1:
            extra_k = np.zeros((M,1))
            P_k = np.concatenate((P_k,extra_k),axis=1)
            P_k_n = np.zeros((1,curr_k+1))
            P_k_n[0,eta_n] = 1
            P_k[n,:] = P_k_n
        else:
            P_k_n = np.zeros((1,curr_k))
            P_k_n[0,eta_n] = 1
            P_k[n,:] = P_k_n
            del rewards[-1]
            del policies[-1]
            del logpolicies[-1]
            del dynets[-1]
            del dyoptims[-1]

        M_k = np.expand_dims(np.sum(P_k, axis=0), axis=1)
        for i in range(len(M_k)):
            if M_k[i] == 0:
                P_k = np.delete(P_k,i,1)
                del rewards[i]
                del policies[i]
                del logpolicies[i]
                del dynets[i]
                del dyoptims[i]
                break

        return P_k, rewards, policies, logpolicies, dynets, dyoptims

    

    def Espected_value_per_n(self, solutions, P_k_reward, P_k_policy):
        M, T = np.shape(self.example_samples)[0:2]
        R = len(self.true_rewards)
        S = len(solutions)
        policy_reward_value = np.zeros((S,R)) 
        for s in range(S):
            for r in range(R):
                policy_reward_value[s,r] = self.model.stdvalue_from_policy(self.true_rewards[r],solutions[s].p)

        EV_n = np.zeros((M))
        for n in range(M):
            reward_idx = np.where(P_k_reward[n,:] == 1)[0][0]
            policy_idx = np.where(P_k_policy[n,:] == 1)[0][0]
            EV_n[n] = policy_reward_value[policy_idx,reward_idx]
        
        EV_argtemp = np.argsort(policy_reward_value, axis=0)
        EV_arg = EV_argtemp[-1,:]

        return EV_n, EV_arg



        












