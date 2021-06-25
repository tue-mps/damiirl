import numpy as np
import numpy.random as random
import math
from utils import Struct, cartcheckleaf, cartaverage



class ObjectWorld(object):


    def __init__(
        self, 
        n = 32, 
        determinism = 0.7, 
        continuous = 0, 
        sample_length = 8, 
        n_samples = 32,
        placement_prob = 0.1,
        discount = 0.9,
        c1 = 2,
        c2 = 2,
        seed = None
    ):

        self.n = n #greed size
        self.states = n**2
        self.actions = 5
        self.determinism = determinism 
        self.continuous = continuous 
        self.sample_length = sample_length 
        self.n_samples = n_samples
        self.placement_prob = placement_prob
        self.discount = discount
        self.c1 = c1
        self.c2 = c2
        self.seed = seed
        self.sa_s, self.sa_p = self.transition()
        self.map1,self.map2,self.c1array,self.c2array = self.map()
        self.feature_data, self.feature_data_r = self.gamefeatures()

    def objectworldrewardtree(self, rule_type):
        x = [0,0,0,0,0]
        y = [-2,-2,-2,-2,-2]
        z = [1,1,1,1,1]
        step = self.c1 + self.c2
        r_tree = Struct()
        r_tree.type = 1,
        r_tree.test = 1+step*2
        r_tree.total_leaves = 3
        r_tree.ltTree = Struct()
        r_tree.ltTree.type = 0
        r_tree.ltTree.index = 1
        if rule_type == 'A':
            r_tree.ltTree.mean = x
        elif rule_type == 'B':
            r_tree.ltTree.mean = z
        elif rule_type == 'C':
            r_tree.ltTree.mean = y
        elif rule_type == 'D':
            r_tree.ltTree.mean = x
        elif rule_type == 'E':
            r_tree.ltTree.mean = y
        elif rule_type == 'F':
            r_tree.ltTree.mean = z
        r_tree.gtTree = Struct()
        r_tree.gtTree.type = 1
        r_tree.gtTree.test = 6
        r_tree.gtTree.total_leaves = 2
        r_tree.gtTree.ltTree = Struct()
        r_tree.gtTree.ltTree.type = 0
        r_tree.gtTree.ltTree.index = 3
        if rule_type == 'A':
            r_tree.gtTree.ltTree.mean = y
        elif rule_type == 'B':
            r_tree.gtTree.ltTree.mean = x
        elif rule_type == 'C':
            r_tree.gtTree.ltTree.mean = z
        elif rule_type == 'D':
            r_tree.gtTree.ltTree.mean = z
        elif rule_type == 'E':
            r_tree.gtTree.ltTree.mean = x
        elif rule_type == 'F':
            r_tree.gtTree.ltTree.mean = y
        r_tree.gtTree.gtTree = Struct()
        r_tree.gtTree.gtTree.type = 0
        r_tree.gtTree.gtTree.index = 2
        if rule_type == 'A':
            r_tree.gtTree.gtTree.mean = z
        elif rule_type == 'B':
            r_tree.gtTree.gtTree.mean = y
        elif rule_type == 'C':
            r_tree.gtTree.gtTree.mean = x
        elif rule_type == 'D':
            r_tree.gtTree.gtTree.mean = y
        elif rule_type == 'E':
            r_tree.gtTree.gtTree.mean = z
        elif rule_type == 'F':
            r_tree.gtTree.gtTree.mean = x
        return r_tree


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

        return sa_s, sa_p

    def map(self):

        random.seed(seed=self.seed)

        map1 = np.zeros((self.n**2,1), int)
        map2 = np.zeros((self.n**2,1), int)
        c1array = [ [] for i in range(self.c1)]
        c2array = [ [] for i in range(self.c1)]


        for round in range(math.ceil(self.c1*0.5)):
            initc1 = (round)*2
            if initc1 + 1 == self.c1:
                prob = self.placement_prob*0.5
                maxc1 = 1
            else:
                prob = self.placement_prob
                maxc1 = 2

            for s in range(self.n**2):
                rd = random.rand(1,1)
                if rd < prob and map1[s] == 0:

                    c1 = initc1 + math.ceil(random.rand(1,1)*maxc1)
                    c2 = math.ceil(random.rand(1,1)*self.c2)
                    map1[s] = c1
                    map2[s] = c2
                    c1array[c1 - 1].append(s)
                    c2array[c2 - 1].append(s)



        return map1, map2, c1array, c2array

    def gamefeatures(self):

        stateadjacency = np.zeros((self.states,self.states))

        for s in range(self.states):
            for a in range(self.actions):
                stateadjacency[s, self.sa_s[s,a,0]] = 1
        

        splittable = np.zeros((self.states, (self.n - 1)*(self.c1 + self.c2)))
        splittablecont = np.zeros((self.states, self.c1 + self.c2))

        for s in range(self.states):

            y = math.ceil((s+1)/self.n) - 1
            x = s + 1 - (y)*self.n - 1

            c1dsq = math.sqrt(2*((self.n)**2))*np.ones((self.c1,1))
            c2dsq = math.sqrt(2*((self.n)**2))*np.ones((self.c2,1))
            
            for i in range(self.c1):
                for j in range(len(self.c1array[i])):
                    cy = math.ceil((self.c1array[i][j] + 1)/self.n) - 1
                    cx = self.c1array[i][j] + 1 - (cy)*self.n - 1
                    d = math.sqrt((cx - x)**2 + (cy - y)**2)
                    c1dsq[i] = min(c1dsq[i],d)

            for i in range(self.c2):
                for j in range(len(self.c2array[i])):
                    cy = math.ceil((self.c2array[i][j] + 1)/self.n) - 1
                    cx = self.c2array[i][j] + 1 - (cy)*self.n - 1
                    d = math.sqrt((cx - x)**2 + (cy - y)**2)
                    c2dsq[i] = min(c2dsq[i],d)


            for d in range(self.n - 1):
                strt = d*(self.c1 + self.c2)
                for i in range(self.c1):
                    splittable[s, strt + i] = c1dsq[i] < d + 1

                strt = d*(self.c1 + self.c2) + self.c1
                for i in range(self.c2):
                    splittable[s, strt + i] = c2dsq[i] < d + 1

            splittablecont[s,:self.c1] = c1dsq[:,0]
            splittablecont[s,-self.c1:] = c2dsq[:,0]

        
        feature_data = Struct()
        feature_data.stateadjacency = stateadjacency
        if self.continuous:
            feature_data.splittable = splittablecont
        else:
            feature_data.splittable = splittable

        feature_data_r = Struct()
        feature_data_r.stateadjacency = stateadjacency
        feature_data_r.splittable = splittable

        return feature_data, feature_data_r

    def gamereward(self, rule_type):

        r_tree = self.objectworldrewardtree(rule_type)

        R_SCALE = 5
        r = cartaverage(r_tree,self.feature_data_r)*R_SCALE

        return r






        


