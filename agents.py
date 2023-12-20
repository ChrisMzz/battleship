from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from skimage.io import imsave
import tqdm
import pdb

mu_plc = 1/3
mu_gss = 1/3
mu_k = 1/3



class Player:
    def __init__(self, 
                 parent1=None,
                 parent2=None,
                 grid_size=10,
                 boats=[2,3,3,4,5],
                 kernel=np.array([[1,1,1],
                                  [1,0,1],
                                  [1,1,1]])/8
                 ):
        
        if parent1 == None and parent2 == None:
            if grid_size**2 < 2*sum(boats) or max(boats)>=grid_size: raise Exception('not enough space for boats')
            self.grid_size = grid_size
            self.kernel = kernel
            self.boats = boats
            self.boat_grid = np.zeros((self.grid_size, self.grid_size))
            self.guess_grid = np.zeros((self.grid_size, self.grid_size))
            self.boat_plc_dstrb = np.ones((self.grid_size,self.grid_size))/self.grid_size**2
            self.boat_gss_dstrb = self.boat_plc_dstrb
            self.boats_shuffle = list(range(len(boats)))
            np.random.shuffle(self.boats_shuffle)
            self.a, self.d = np.random.rand(2)*2-1
        else:
            self.kernel = parent1.kernel
            self.boats = parent1.boats
            self.grid_size = parent1.grid_size
            self.boat_grid = np.zeros((self.grid_size, self.grid_size))
            self.guess_grid = np.zeros((self.grid_size, self.grid_size))
            self.a, self.d = max(parent1.a,parent2.a)*(1+np.random.rand()*0.1), max(parent1.d,parent2.d)*(1+np.random.rand()*0.1)
            self.boat_plc_dstrb = (parent1.boat_plc_dstrb + parent2.boat_plc_dstrb)/2
            has_mutation = np.random.rand()
            if has_mutation < mu_plc:
                i,j = np.random.randint(0, self.grid_size, 2)
                poi = np.zeros((self.grid_size, self.grid_size))
                poi[i,j] = 1
                self.boat_plc_dstrb += convolve(poi, np.array([[1,1,1],
                                                               [1,4,1],
                                                               [1,1,1]])/12)
                self.boat_plc_dstrb /= np.sum(self.boat_plc_dstrb)
            self.boat_gss_dstrb = (parent1.boat_gss_dstrb + parent2.boat_gss_dstrb)/2
            has_mutation = np.random.rand()
            if has_mutation < mu_gss:
                i,j = np.random.randint(0, self.grid_size, 2)
                poi = np.zeros((self.grid_size, self.grid_size))
                poi[i,j] = 1
                self.boat_gss_dstrb += convolve(poi, np.array([[1,1,1],
                                                               [1,4,1],
                                                               [1,1,1]])/12)
                self.boat_gss_dstrb /= np.sum(self.boat_gss_dstrb)
            
            self.boats_shuffle = [parent1.boats_shuffle, parent2.boats_shuffle][np.random.randint(0,2)]
            has_mutation = np.random.rand()
            if has_mutation < mu_k: m_kernel = np.array(np.random.randint(-2,3, size=(3,3)), dtype=np.float64)
            else: m_kernel = np.zeros((3,3))
            if np.sum(m_kernel) != 0: 
                m_kernel /= np.sum(m_kernel)
                self.kernel = (parent1.kernel + parent2.kernel + m_kernel)/3
            else: 
                self.kernel = (parent1.kernel + parent2.kernel + m_kernel)/2
            
        
        self.place_boats()
        
    
    def place_boats(self):
        for s in self.boats_shuffle:
            boat = self.boats[s]
            tries = 0
            boat_placed = False
            while not boat_placed:
                line = np.random.choice(range(self.grid_size), p=[sum(self.boat_plc_dstrb[i,:]) for i in range(self.grid_size)])
                column = np.random.choice(range(self.grid_size), p=self.boat_plc_dstrb[line]/sum(self.boat_plc_dstrb[line]))
                # slicing this way NEVER crashes, which is neat
                
                directions = list(range(4))
                np.random.shuffle(directions)
                i = 0
                while i < 4:
                    direction = directions[i]
                    if direction == 0:
                        if np.all(self.boat_grid[line,column:][:boat] == 0) and not len(self.boat_grid[line,column:]) < boat:
                            self.boat_grid[line,column:][:boat] = 1
                            boat_placed = True
                            break
                    if direction == 1:
                        if np.all(self.boat_grid[line,:column+1][-boat:] == 0) and not len(self.boat_grid[line,:column+1]) < boat:
                            self.boat_grid[line,:column+1][-boat:] = 1
                            boat_placed = True
                            break
                    if direction == 2:
                        if np.all(self.boat_grid[line:,column][:boat] == 0) and not len(self.boat_grid[line:,column]) < boat:
                            self.boat_grid[line:,column][:boat] = 1
                            boat_placed = True
                            break
                    if direction == 3:
                        if np.all(self.boat_grid[:line+1,column][-boat:] == 0) and not len(self.boat_grid[:line+1,column]) < boat:
                            self.boat_grid[:line+1,column][-boat:] = 1
                            boat_placed = True
                            break
                    i += 1
                    
                tries += 1
                if tries > 5*self.grid_size**2: break
                
                
    def guess(self, other:Player):
        dstrb = ( self.boat_gss_dstrb + convolve(self.guess_grid==2, self.kernel) )*(self.guess_grid==0)
        dstrb /= np.sum(dstrb)
        if np.isnan(dstrb).any(): 
            dstrb = (np.ones((self.grid_size,self.grid_size))*(self.guess_grid==0))
            dstrb /= np.sum(dstrb)
        line = np.random.choice(range(self.grid_size), p=[sum(dstrb[i,:]) for i in range(self.grid_size)])
        column = np.random.choice(range(self.grid_size), p=dstrb[line]/sum(dstrb[line]))
        if other.boat_grid[line,column] == 1: self.guess_grid[line,column] = 2
        else: self.guess_grid[line,column]  = 1
        return (other.boat_grid[line,column] == 1), (line, column)
            

def game(black:Player, white:Player):
    grid = [np.c_[black.boat_grid+2*white.guess_grid+1, np.zeros((black.grid_size,black.grid_size)), white.boat_grid+2*black.guess_grid+1]]
    while True: # I'm sorry T.T
        hit, pos = white.guess(black)
        if hit:
            hit_grid = np.zeros((white.grid_size, white.grid_size))
            hit_grid[pos[0], pos[1]] = 1
            hit_grid_a = convolve(hit_grid, black.a*np.array([[1,1,1],
                                                              [1,4,1],
                                                              [1,1,1]])/12)
            hit_grid_d = convolve(hit_grid, white.d*np.array([[1,1,1],
                                                              [1,4,1],
                                                              [1,1,1]])/12)
            white.boat_gss_dstrb += hit_grid_a
            white.boat_gss_dstrb *= (white.boat_gss_dstrb>0)
            white.boat_gss_dstrb /= np.sum(white.boat_gss_dstrb)
            black.boat_plc_dstrb -= hit_grid_d
            black.boat_plc_dstrb *= (black.boat_plc_dstrb>0)
            black.boat_plc_dstrb /= np.sum(black.boat_plc_dstrb)
        grid.append(np.c_[black.boat_grid+2*white.guess_grid+1, np.zeros((black.grid_size,black.grid_size)), white.boat_grid+2*black.guess_grid+1])
        if np.sum(white.guess_grid==2) == np.sum(black.boat_grid): return white, np.array(grid)
        
        hit, pos = black.guess(white)
        if hit:
            hit_grid = np.zeros((black.grid_size, black.grid_size))
            hit_grid[pos[0], pos[1]] = 1
            hit_grid_a = convolve(hit_grid, black.a*np.array([[1,1,1],
                                                              [1,4,1],
                                                              [1,1,1]])/12)
            hit_grid_d = convolve(hit_grid, white.d*np.array([[1,1,1],
                                                              [1,4,1],
                                                              [1,1,1]])/12)
            black.boat_gss_dstrb += hit_grid_a
            black.boat_gss_dstrb *= (black.boat_gss_dstrb>0)
            black.boat_gss_dstrb /= np.sum(black.boat_gss_dstrb)
            white.boat_plc_dstrb -= hit_grid_d
            white.boat_plc_dstrb *= (white.boat_plc_dstrb>0)
            white.boat_plc_dstrb /= np.sum(white.boat_plc_dstrb)
        grid.append(np.c_[black.boat_grid+2*white.guess_grid+1, np.zeros((black.grid_size,black.grid_size)), white.boat_grid+2*black.guess_grid+1])
        if np.sum(black.guess_grid==2) == np.sum(white.boat_grid): return black, np.array(grid)
        
    



if __name__=='__main__':
    
    pop_size = 100 # multiple of 4


    next_generation = [Player(boats=[3,4,5,5,8,10], grid_size=20) for _ in range(pop_size)]
    #pdb.set_trace()
    
    progress_bar = tqdm.trange(10)
    for n in progress_bar:
        black_bwoks, white_bwoks = next_generation[:pop_size//2+1], next_generation[pop_size//2:]
        winners_and_games = [game(black, white) for black, white in zip(black_bwoks, white_bwoks)]
        winners = [winners_and_games[i][0] for i in range(len(winners_and_games))]
        games = [winners_and_games[i][1] for i in range(len(winners_and_games))]
        progress_bar.set_postfix({"average_length":np.mean([len(games[i]) for i in range(len(games))])})
        #pdb.set_trace()
        next_generation = []
        for _ in range(4):
            np.random.shuffle(winners)
            for black, white in zip(winners[:pop_size//4+1], winners[pop_size//4:]):
                next_generation.append(Player(black, white))
    
    imsave("replay_400.tif", game(*next_generation[:2])[1])
    
    #pdb.set_trace()



