import random
import time
import math
from copy import deepcopy

from Game4InLine import Game4InLine as G4


class Node:
    '''
    class that defines the nodes for the MCTS tree
    '''
    def __init__(self, game: G4, parent=None):
        '''
        game is the game state, parent is the father node
        childs is started as empty so that we can differenciate from an explored node or no
        '''
        self.game=deepcopy(game)
        self.parent=parent
        self.visited=0
        self.wins=0
        self.childs = [] # formato [[node,col]]

    def set_childs(self):
        '''
        set the childs for the node based on the legal_moves from the state os the board
        appends a list to the list with the format ->  [child_node, col]
        child_node is the possible game state from the current node made by playing the column -> col 
        '''
        poss_moves = self.game.legal_moves()
        for col in poss_moves:
            state = deepcopy(self.game)
            self.childs.append([Node(state.play(col), parent=self),col])

    def UCB1(self):
        '''
        funcion that calculates the UCB1 for the node 
        '''
        if self.visited == 0:
            return float('inf')

        return ((self.wins/self.visited) + math.sqrt(2) * math.sqrt(2*math.log(self.parent.visited)/self.visited))

    def max_UCB(self):
        '''
        return the max UCB1 from the childs of the node
        used to know the best nodes at the explore/selection phase
        '''
        max_val = (self.childs[0][0]).UCB1()
        for i in range (1,len(self.childs)):
            max_val = max(max_val, (self.childs[i][0]).UCB1())
        return max_val


class MCTS:
    '''
    class that defines a tree for MCTS
    '''

    def __init__(self, root: G4):
        self.root=Node(deepcopy(root))
        self.run_time = 0
        self.simulations = 0
    

    def search(self, time_limit, limit_simulations=22500):
        '''
        main function to execute the MCTS
        based on multiple tests, we found that MCTS works better with 5 seconds limit or 22500 iterations
        so we end the search when one of this conditions are broken
        '''
        start_time = time.time()
        simulations = 0
        while time.time() - start_time < time_limit and simulations < limit_simulations:
            node, col = self.selection()
            result = self.simulate(deepcopy(node.game),col)
            self.back_propagate(node,result)
            simulations += 1


        self.run_time = time.time() - start_time
        self.simulations = simulations


    def selection(self):
        '''
        we start on root node and will go to the childs of the node and select the best case to expand/simulate based on the UCB1 given
        and we repeat until we reach a leaf or the node as not been visited
        if we can expand the node we select randomly from a childs
        '''
        node = self.root
        col_child = 0 # needed due to who we set up the isFinished() funcion used in expand()

        while len(node.childs) != 0:
            max_ucb = node.max_UCB()
            max_nodes = [n for n in node.childs if n[0].UCB1() == max_ucb]
            best_child = random.choice(max_nodes)
            node = best_child[0]
            col_child = best_child[1]

            if node.visited == 0:
                return node, col_child
            
        if self.expand(node,col_child):
            random_child = random.choice(node.childs)
            node = random_child[0]
            col_child = random_child[1]

        return node, col_child

    def expand(self, node: Node, col: int):
        '''
        returns False if the game is over or True after we add childs
        if node is not a end for the game we add the childs
        '''
        if node.game.isFinished(col):
            return False
        
        node.set_childs()
        return True

    def simulate(self, node_state: G4, last_played: int):
        '''
        randomly select a valid column to play and repeats until the game is over
        return 0 if the game is lose or draw and 1 if win, based on the last piece played
        '''
        res = node_state.isFinished(last_played) #in case the node given is already finished
        if res:
            return res if res == 2 else 0 if node_state.turn%2==0 else 1

        while True:
            col = random.choice(node_state.legal_moves())
            node_state.play(col)
            res = node_state.isFinished(col)
            if res:
                return res if res == 2 else 0 if node_state.turn%2==0 else 1

    def back_propagate(self, node: Node, result):
        '''
        we go from the given node to the root of the tree and do the respective alterations to the data
        when player 1 win we give to all the node where player 1 played +1 on wins and +0 for else
        for when player 2 win we follow the same logic

        result = 2 if draw, =1 if 'X' won and =0 if 'O' won
        logo quando result == node.game.turn vai dar reward 1 pois foi o jogador que ganhou a simulacao
        '''
        while node != None:
            if result == node.game.turn: #when the winner is the same as the last player on the curr node 
                reward = 1
            else: reward = 0
            node.visited += 1
            node.wins += reward
            node = node.parent


    def best_move(self):
        '''
        after the search() we select the best child from the root with this funcion
        we also print the win_rate for each child to visualize the data and the choise made from the algoritm
        '''
        childs = self.root.childs
        if len(childs)==0:
            return random.choice(self.root.game.legal_moves())
        node = childs[0][0]
        best_col = childs[0][1]
        max_win_rate = node.wins/node.visited
        print(f"win/visited: {max_win_rate:.4f} col: {best_col+1}")

        for i in range (1,len(childs)):
            node = childs[i][0]
            max_win_rate_temp = node.wins/node.visited
            print(f"win/visited: {max_win_rate_temp:.4f} col: {childs[i][1]+1}")
            if max_win_rate < max_win_rate_temp:
                max_win_rate = max_win_rate_temp
                best_col = childs[i][1]

        return best_col
    
    def statistic(self):
        ''' returns the number of simulations and the run time for the search taken '''
        return self.simulations, self.run_time