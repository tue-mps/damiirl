import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import math
from matplotlib.gridspec import GridSpec
plt.rcParams["font.family"] = "serif"
plt.rcParams.update({'font.size': 12})
matplotlib.use('agg')


class Drawing(object):

    def __init__(
    self, 
    game,
    true_rewards,
    rewardssquence,
    model, 
    model_solutions,
    example_samples, 
    mirl_solutions,
    path
    ):

        self.game = game
        self.true_rewards = true_rewards
        self.rewardssquence = rewardssquence
        self.model = model
        self.model_solutions = model_solutions
        self.example_samples = example_samples
        self.mirl_solutions = mirl_solutions
        self.path = path
        self.g = self.visible_examples()
        self.worlddraw()
        


    def visible_examples(self):

        Eo = np.zeros((self.game.states))

        N = np.shape(self.example_samples)[0]
        T = np.shape(self.example_samples)[1]

        for i in range(N):
            for t in range(T):
                Eo[self.example_samples[i][t][0]] = 1
        
        g = np.ones((self.game.states))*0.5 + Eo*0.5
        g = np.ones((self.game.states))

        return g

    def worlddraw(self):

        fig = plt.figure(constrained_layout=True)
        

        gs = GridSpec(2, 6, figure=fig)

        Gr = len(self.model_solutions)
        Pr = len(self.mirl_solutions)


        for gr in range(Gr):
            ax = fig.add_subplot(gs[0, gr])
            r_true = self.true_rewards[gr]
            p_true = self.model_solutions[gr].p
            self.rewarddraw(r_true, ax, 'True \n reward function '+str(gr+1))
            self.policydraw(p_true, ax)
            self.objectdraw(ax)


        for pr in range(Pr):

            ax = fig.add_subplot(gs[1, pr])
            r_pred_mirl = self.mirl_solutions[pr].r
            p_pred_mirl = self.mirl_solutions[pr].p
            self.rewarddraw(r_pred_mirl, ax, 'Predicted \n reward function '+str(pr+1))
            self.policydraw(p_pred_mirl, ax)
            self.objectdraw(ax)
                


        
        fig.set_size_inches((18, 10), forward=False)
        plt.savefig(self.path, dpi=500)

    def rewarddraw(self,r, ax, name):

        n = self.game.n
        maxr = np.max(r)
        minr = np.min(r)
        rngr = maxr - minr




        xticks = np.arange(0,n+1,1)
        yticks = np.arange(0,n+1,1)

        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect(1)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis='both', which='both', length=0)
        ax.grid(True, color='black')
        ax.set_title(name)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        for y in range(n):
            for x in range(n):

                if rngr == 0:
                    v = 0
                else:
                    temp1 = r[y*n+x,:]
                    temp2 = np.mean(temp1)
                    v = ( temp2 - minr)/rngr

                color = np.array([v,v,v])
                temp = np.maximum(np.zeros((3)), color)
                color = np.minimum(np.ones((3)),temp)
                left, bottom, width, height = (x, y, 1, 1)
                rect = plt.Rectangle((left, bottom), width, height,facecolor=color)
                ax.add_patch(rect)

    def policydraw(self, p, ax):

        p = np.argmax(p, axis=1)

        n = self.game.n
        g = self.g
        for y in range(n):
            for x in range(n):
                s = y*n + x
                a = p[s]
                color = np.array([g[s],g[s],g[s]])
                w = 1

                if a == 0:
                    
                    nx = x + 0.5
                    ny = y + 0.5
                    xy = np.array([
                        [nx + 0.25,ny],
                        [nx, ny - 0.25],
                        [nx - 0.25, ny],
                        [nx, ny + 0.25]
                    ])
                    poly = plt.Polygon(xy,facecolor=color, linewidth=w, edgecolor='black')
                    ax.add_patch(poly)

                else:
                    if a == 4:
                        nx = x - 1
                        ny = y
                    elif a == 3:
                        nx = x
                        ny = y - 1
                    elif a == 2:
                        nx = x + 1
                        ny = y
                    elif a == 1:
                        nx = x
                        ny = y + 1

                    vec = np.array([(nx - x)*0.25,(ny - y)*0.25])
                    normal1 = np.array([vec[1],-vec[0]])
                    normal2 = -normal1
                    xy = np.array([
                        [x + 0.5 + vec[0], y + 0.5 + vec[1]],
                        [x + 0.5 + normal1[0], y + 0.5 + normal1[1]],
                        [x + 0.5 + normal2[0], y + 0.5 + normal2[1]]
                    ])
                    poly = plt.Polygon(xy,facecolor=color, linewidth=w, edgecolor='black')
                    ax.add_patch(poly)

    def objectdraw(self, ax):
        n = self.game.n

        c1array = self.game.c1array

        shapeColors = pl.cm.jet(np.linspace(0,1,self.game.c1 + self.game.c2))

        for i in range(len(c1array)):
            for j in range(len(c1array[i])):
                
                s = c1array[i][j] + 1
                c1 = i
                c2 = self.game.map2[s-1] - 1
                y = math.ceil(s/n) - 1
                x = s - (y)*n - 1
                xy = np.array([x + 0.5, y + 0.5])
                shcol = np.squeeze(shapeColors[self.game.c1+c2,:], axis=0)
                ecol = shapeColors[c1,:]
                ccl = plt.Circle(xy, 0.12, facecolor=shcol,linewidth=2,edgecolor=ecol)
                ax.add_patch(ccl)


            
class Drawing_Prediction(object):

    def __init__(
    self, 
    game,
    pred_sample,
    gt_sample,
    obs_sample,
    ):

        self.game = game
        self.pred_sample = pred_sample
        self.gt_sample = gt_sample
        self.obs_sample = obs_sample
        self.worlddraw()


    def worlddraw(self):



        _, ax = plt.subplots(1,3)


        self.obspathdraw(-self.obs_sample, ax[0])
        self.obspathdraw(-self.gt_sample, ax[1])
        self.obspathdraw(-self.pred_sample, ax[2])
        self.objectdraw(ax[0])
        self.objectdraw(ax[1])
        self.objectdraw(ax[2])

                


        plt.show()

    def obspathdraw(self,path, ax):

        n = self.game.n
        maxpath = np.max(path)
        minpath = np.min(path)
        rngpath = maxpath - minpath




        xticks = np.arange(0,n+1,1)
        yticks = np.arange(0,n+1,1)

        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect(1)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis='both', which='both', length=0)
        ax.grid(True, color='black')

        for y in range(n):
            for x in range(n):

                if rngpath == 0:
                    v = 0
                else:
                    temp1 = path[y*n+x,:]
                    temp2 = np.mean(temp1)
                    v = ( temp2 - minpath)/rngpath

                color = np.array([v,v,v])
                temp = np.maximum(np.zeros((3)), color)
                color = np.minimum(np.ones((3)),temp)
                left, bottom, width, height = (x, y, 1, 1)
                rect = plt.Rectangle((left, bottom), width, height,facecolor=color)
                ax.add_patch(rect)

    def policydraw(self, p, ax):

        p = np.argmax(p, axis=1)

        n = self.game.n
        g = self.g
        for y in range(n):
            for x in range(n):
                s = y*n + x
                a = p[s]
                color = np.array([g[s],g[s],g[s]])
                w = 1

                if a == 0:
                    
                    nx = x + 0.5
                    ny = y + 0.5
                    xy = np.array([
                        [nx + 0.25,ny],
                        [nx, ny - 0.25],
                        [nx - 0.25, ny],
                        [nx, ny + 0.25]
                    ])
                    poly = plt.Polygon(xy,facecolor=color, linewidth=w, edgecolor='black')
                    ax.add_patch(poly)

                else:
                    if a == 4:
                        nx = x - 1
                        ny = y
                    elif a == 3:
                        nx = x
                        ny = y - 1
                    elif a == 2:
                        nx = x + 1
                        ny = y
                    elif a == 1:
                        nx = x
                        ny = y + 1

                    vec = np.array([(nx - x)*0.25,(ny - y)*0.25])
                    normal1 = np.array([vec[1],-vec[0]])
                    normal2 = -normal1
                    xy = np.array([
                        [x + 0.5 + vec[0], y + 0.5 + vec[1]],
                        [x + 0.5 + normal1[0], y + 0.5 + normal1[1]],
                        [x + 0.5 + normal2[0], y + 0.5 + normal2[1]]
                    ])
                    poly = plt.Polygon(xy,facecolor=color, linewidth=w, edgecolor='black')
                    ax.add_patch(poly)

    def objectdraw(self, ax):
        n = self.game.n

        c1array = self.game.c1array

        shapeColors = pl.cm.jet(np.linspace(0,1,self.game.c1 + self.game.c2))

        for i in range(len(c1array)):
            for j in range(len(c1array[i])):
                
                s = c1array[i][j] + 1
                c1 = i
                c2 = self.game.map2[s-1] - 1
                y = math.ceil(s/n) - 1
                x = s - (y)*n - 1
                xy = np.array([x + 0.5, y + 0.5])
                shcol = np.squeeze(shapeColors[self.game.c1+c2,:], axis=0)
                ecol = shapeColors[c1,:]
                ccl = plt.Circle(xy, 0.12, facecolor=shcol,linewidth=2,edgecolor=ecol)
                ax.add_patch(ccl)

