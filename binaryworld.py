import numpy as np
import numpy.random as random
import math
from utils import Struct, cartcheckleaf, cartaverage



class BinaryWorld(object):


    def __init__(
        self, 
        n = 32, 
        determinism = 0.7, 
        sample_length = 8, 
        n_samples = 32,
        discount = 0.9,
        seed = None
    ):

        self.n = n #greed size
        self.states = n**2
        self.actions = 5
        self.determinism = determinism 
        self.sample_length = sample_length 
        self.n_samples = n_samples
        self.placement_prob = 0.5
        self.discount = discount
        self.c1 = 2
        self.c2 = 2
        self.seed = seed
        self.sa_s, self.sa_p = self.transition()
        self.map1, self.map2, self.c1array, self.c2array = self.map()
        self.feature_data = self.gamefeatures()



    def transition(self):

        sa_s = np.zeros((self.n**2,5,5), int)
        sa_p = np.zeros((self.n**2,5,5))

        for y in range(self.n):
            for x in range(self.n):

                s = y*self.n + x + 1
                successors = np.zeros((1,1,5))
                successors[0,0,0] = s - 1 
                successors[0,0,1] = (min(self.n,y+2)-1)*self.n + x + 1 - 1
                successors[0,0,2] = y*self.n + min(self.n,x+2) - 1
                successors[0,0,3] = (max(1,y)-1)*self.n+x+1 - 1
                successors[0,0,4] = y*self.n+max(1,x) - 1
                sa_s[s-1,:,:] = np.tile(successors, (1, 5, 1))
                sa_p[s-1,:,:] = np.reshape(
                    np.eye(5)*self.determinism + (np.ones(5) - np.eye(5))*((1 - self.determinism)/4), 
                    (1, 5, 5)
                )

        #print(sa_s)

        return sa_s, sa_p

    def map(self):

        random.seed(seed=self.seed)

        map1 = np.zeros((self.n**2,1), int)
        c1array = [ [] for i in range(self.c1)]


        prob = self.placement_prob

        for s in range(self.n**2):
            rd = random.rand(1,1)
            #print(rd)
            if rd < prob:
                c = 0
            else:
                c = 1
            map1[s] = c
            c1array[c].append(s)

        map2 = np.copy(map1)
        c2array = np.copy(map2)
        
        return map1, map2, c1array, c2array

    def gamefeatures(self):
        

        splittable = np.zeros((self.states, 9))

        for s in range(self.states):

            y = math.ceil((s+1)/self.n) - 1
            x = s + 1 - (y)*self.n - 1

            indx = np.zeros(9, int)
            indy = np.zeros(9, int)
            indall = np.arange(9)

            indx[0] = x - 1
            indx[1] = x - 1
            indx[2] = x - 1
            indx[3] = x 
            indx[4] = x 
            indx[5] = x 
            indx[6] = x + 1
            indx[7] = x + 1
            indx[8] = x + 1
            indy[0] = y - 1
            indy[1] = y 
            indy[2] = y + 1
            indy[3] = y - 1
            indy[4] = y 
            indy[5] = y + 1
            indy[6] = y - 1
            indy[7] = y 
            indy[8] = y + 1

            indsel = indy * self.n + indx

            indselin = []
            for i in range(9):
                if indx[i] >= 0 and indx[i] < self.n  and indy[i] >= 0 and indy[i] < self.n:
                    indselin.append(indall[i])
            
            indselin = np.array(indselin)
            temp = indsel[indselin]
            splittable[s,indselin] = self.map1[temp,0]

        feature_data = Struct()
        feature_data.splittable = np.sort(splittable, axis=1)


        return feature_data

    def gamereward(self, rule_type):

        r = np.zeros((self.states,1))
        feature_data = self.feature_data.splittable
        x = 1
        y = -2
        z = 0
        for s in range(self.states):
            if np.sum(feature_data[s,:]) == 4:
                if rule_type == 'A':
                    r[s] = x
                elif rule_type == 'B':
                    r[s] = z
                elif rule_type == 'C':
                    r[s] = y
                elif rule_type == 'D':
                    r[s] = x
                elif rule_type == 'E':
                    r[s] = y
                elif rule_type == 'F':
                    r[s] = z
            elif np.sum(feature_data[s,:]) == 5:
                if rule_type == 'A':
                    r[s] = y
                elif rule_type == 'B':
                    r[s] = x
                elif rule_type == 'C':
                    r[s] = z
                elif rule_type == 'D':
                    r[s] = z
                elif rule_type == 'E':
                    r[s] = x
                elif rule_type == 'F':
                    r[s] = y
            else:
                if rule_type == 'A':
                    r[s] = z
                elif rule_type == 'B':
                    r[s] = y
                elif rule_type == 'C':
                    r[s] = x
                elif rule_type == 'D':
                    r[s] = y
                elif rule_type == 'E':
                    r[s] = z
                elif rule_type == 'F':
                    r[s] = x

        R_SCALE = 5
        r = np.tile(r*R_SCALE, (1,self.actions))

        return r






        


