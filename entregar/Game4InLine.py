from copy import deepcopy
import joblib
import pandas as pd
import random

#Human start and uses 'X' (player 1)
#AI go second and uses 'O' (player 2)

class Game4InLine:

    def __init__ (self,row,col):
        '''
        initialize the game, board and all other components to be able to play
        '''
        self.decision_tree = joblib.load('arvore_4connected.pkl') # arvore de decisao criada pelo algoritmo
        self.rows = row
        self.cols = col
        self.board = [['b' for _ in range(col)] for _ in range(row)] #matrix representing the board, initialized full with '-'
        self.placed = [0 for _ in range(col)] #store the num of pieces per column, initialized with 0s
        self.pieces = ['x','o'] # different pieces
        self.turn = 0 # to know next player, switch between 0 and 1
        self.round = 0 


    def legal_moves(self):
        '''
        returns a list with the columns that are possible to play,
        this is, that are not full
        '''
        legal=[]

        for i in range(self.cols):
            if self.placed[i]<self.rows: #placed[i]<rows means that the number of pieced placed in the row-i is less than the max pieced that are possible to place
                legal.append(i)
        
        return legal


    def childs(self):
        '''
        this funcion returns a list for the possible childs based on the current state of the board
        the list is made of lists with the format -> [child: Game4InLine, col: int]
        where child is the new game made from the current and played at the col selected
        '''
        moves=self.legal_moves() # returns the possible moves
        children = []

        for col in moves:
            temp=deepcopy(self)
            children.append([temp.play(col),col])

        return children


    def play(self,col:int): #funcion to place pieces based on turn and column
        '''
        it is guaranteed that the col given is not full

        given a col it will place the piece, X or O based on turn
        the piece will be placed at the bottom of the column
        and updated all the data regarding turn, round and placed[]
        '''  
        self.board[self.rows-self.placed[col]-1][col]=self.pieces[self.turn]
        self.placed[col]+=1
        self.round+=1
        self.turn = 1 if self.turn%2==0 else 0 #change turn
        return self


    def isFinished(self,col):
        '''
        return 2 if game is a draw, True if last move was a winning one , False to keep playing
        (we return True for win and 2 for draw because 2!=True but "if isFinished()" will be considered true if the return is 2 and we use it for diferenciate from draw or win)

        based on the game and the column played last we analyze if there is a sequence of 4(or more) of the same type of piece placed
        we start at the position of the last played piece and check vertical, horizontal and both diagonals
        if there is no sequence of 4(or more) but the board is full (when round == rows*cols) we return 2 for a draw
        '''
        played = self.pieces[self.turn-1]
        row = self.rows - max(self.placed[col],1)
    
        #Check Vertical    |
        consecutive = 1  # |
        tmprow = row     # |
        while tmprow+1 < self.rows and self.board[tmprow+1][col] == played:
            consecutive+=1
            tmprow+=1
        if consecutive >= 4:
            return True


        # Check horizontal  ----
        consecutive = 1
        tmpcol = col
        while tmpcol+1 < self.cols and self.board[row][tmpcol+1] == played:
            consecutive += 1
            tmpcol += 1
        tmpcol = col
        while tmpcol-1 >= 0 and self.board[row][tmpcol-1] == played:
            consecutive += 1
            tmpcol -= 1
        
        if consecutive >= 4:
            return True


        # Check diagonal   1          \
        consecutive = 1             #  \
        tmprow = row                 #  \
        tmpcol = col                  #  \
        while tmprow+1 < self.rows and tmpcol+1 < self.cols and self.board[tmprow+1][tmpcol+1] == played:
            consecutive += 1
            tmprow += 1
            tmpcol += 1
        tmprow = row
        tmpcol = col
        while tmprow-1 >= 0 and tmpcol-1 >= 0 and self.board[tmprow-1][tmpcol-1] == played:
            consecutive += 1
            tmprow -= 1
            tmpcol -= 1

        if consecutive >= 4:
            return True


        # Check diagonal   2          /
        consecutive = 1           #  /
        tmprow = row             #  /
        tmpcol = col            #  /
        while tmprow-1 >= 0 and tmpcol+1 < self.cols and self.board[tmprow-1][tmpcol+1] == played:
            consecutive += 1
            tmprow -= 1
            tmpcol += 1
        tmprow = row
        tmpcol = col
        while tmprow+1 < self.rows and tmpcol-1 >= 0 and self.board[tmprow+1][tmpcol-1] == played:
            consecutive += 1
            tmprow += 1
            tmpcol -= 1

        if consecutive >= 4:
            return True


        # Check for draw
        if(self.round==(self.rows*self.cols)): #maybe change this to other function
            return 2

        return False


    def heuristic_extra(game,col):
        '''
        this heuristic was made to make A* more defensive 
        where it will prioritize not losing rather than getting max point from the given heuristic for the project

        it gives points based on col played and the player turn
        if it is a win move it will give -512 for O and 512 for X as the given heuristic
        but if it has a chance to defend from a lose, for example: O played and he made a play that got a sequence of 3-X and 1-O
        it will consider that it was a good defence move because it made impossible for the X to win next turn

        the point are given as follow:
        -500 for 3 Xs and 1 O and it is O turn
        -100 for 2 Xs and 1 O and it is O turn
        100 for 2 Os and 1 X and it is X turn
        500 for 3 Os and 1 X and it is X turn
        0 for else
        '''
        row = game.rows-game.placed[col]
    
        #caso for uma jogada para ganhar
        if game.isFinished(col):
            return -512 if game.turn%2==0 else 512


        #caso for uma jogada para nao perder
        points=0
        #horizontal
        for tpmcol in range(max(0,col-3),col+1):
            if tpmcol+3>game.cols-1: continue
            count_X=0
            count_O=0
            for j in range(4):
                if (game.board[row][tpmcol+j] == "x"):
                    count_X+=1
                if (game.board[row][tpmcol+j] == "o"):
                    count_O+=1
            
            #dar pontos
            h_value = getPoints_extra(game,count_X,count_O)
            if abs(h_value) == 500:
                return -500 if game.turn%2==0 else 500
            elif h_value != 0:
                points=h_value
            
       

        #vertical
        count_X=0
        count_O=0
        for i in range(0,min(4,game.rows-row)):
            if (game.board[row+i][col] == "x"):
                count_X+=1
            if (game.board[row+i][col] == "o"):
                count_O+=1

        #dar pontos
        h_value = getPoints_extra(game,count_X,count_O)
        if abs(h_value) == 500:
            return -500 if game.turn%2==0 else 500
        elif h_value != 0:
            points=h_value              
      
       

        #diagonal 1
        tmpcol = col
        tmprow = row
        i=0
        while(i<3 and tmprow>0 and tmpcol>0):
            tmpcol-=1
            tmprow-=1
            i+=1       
        while i>=0 and tmprow+3<game.rows and tmpcol+3<game.cols:
            i-=1
            count_X=0
            count_O=0
            for h in range(4):
                if (game.board[tmprow+h][tmpcol+h] == "x"):
                    count_X+=1
                if (game.board[tmprow+h][tmpcol+h] == "o"):
                    count_O+=1

            h_value = getPoints_extra(game,count_X,count_O)
            if abs(h_value) == 500:
                return -500 if game.turn%2==0 else 500
            elif h_value != 0:
                points=h_value
            
            tmpcol+=1
            tmprow+=1


        #diagonal 2
        tmpcol = col
        tmprow = row
        i=0
        while i<3 and tmprow<game.rows-1 and tmpcol>0:
            tmpcol-=1
            tmprow+=1
            i+=1       

        while i>=0 and tmprow-3>=0 and tmpcol+3<game.cols:
            i-=1
            count_X=0
            count_O=0
            for h in range(4):
                if (game.board[tmprow-h][tmpcol+h] == "x"):
                    count_X+=1
                if (game.board[tmprow-h][tmpcol+h] == "o"):
                    count_O+=1

            h_value = getPoints_extra(game,count_X,count_O)
            if abs(h_value) == 500:
                return -500 if game.turn%2==0 else 500
            elif h_value != 0:
                points=h_value 

            tmpcol+=1
            tmprow-=1 
    
    
        return abs(points)*(-1) if game.turn%2==0 else abs(points)


    def heuristic_points(game,col):
        '''
        this is the given heuristic for the project, it takes col as an input but it doesn't use it, this happen because the other heuristic needs col
        and we use a lambda funcion on our A* that takes the sum of both heuristisc

        the point are given as follow:
        -50 for three Os, no Xs,
        -10 for two Os, no Xs,
        - 1 for one O, no Xs,
        0 for no tokens, or mixed Xs and Os,
        1 for one X, no Os,
        10 for two Xs, no Os,
        50 for three Xs, no Os.

        and depending on whose turn is to play (+16 for X, -16 for O)
        '''

        points = 16 if game.pieces[game.turn] == 'x' else -16
        #horizontal
        for i in range(game.rows):
            for j in range(game.cols-3):
                count_X=0
                count_O=0
                for h in range(j,j+4):
                    if (game.board[i][h] == "x"):
                        count_X+=1
                    if (game.board[i][h] == "o"):
                        count_O+=1

                h_value = getPoints(count_X,count_O)
                if abs(h_value) == 512:
                    return h_value
                else:
                    points+=h_value
        
        #vertical
        for i in range(game.cols):
            for j in range(game.rows-3):
                count_X=0
                count_O=0                
                for h in range(j,j+4):
                    if (game.board[h][i] == "x"):
                        count_X+=1
                    if (game.board[h][i] == "o"):
                        count_O+=1

                h_value = getPoints(count_X,count_O)
                if abs(h_value) == 512:
                    return h_value
                else:
                    points+=h_value

        #diagonal 1
        for i in range(game.rows-3):
            for j in range(game.cols-3):
                count_X=0
                count_O=0  
                for h in range(4):
                    if (game.board[i+h][j+h] == "x"):
                        count_X+=1
                    if (game.board[i+h][j+h] == "o"):
                        count_O+=1

                h_value = getPoints(count_X,count_O)
                if abs(h_value) == 512:
                    return h_value
                else:
                    points+=h_value                    

        #diagonal 2
        for i in range(game.rows-3):
            for j in range(3,game.cols):
                count_X=0
                count_O=0  
                for h in range(4):
                    if (game.board[i+h][j-h] == "x"):
                        count_X+=1
                    if (game.board[i+h][j-h] == "o"):
                        count_O+=1

                h_value = getPoints(count_X,count_O)
                if abs(h_value) == 512:
                    return h_value
                else:
                    points+=h_value

        return points


    def A_star(self,dt):
        '''
        As in this project our A* only looks for its next best play without going in depht for a possible move from its the oponent we don't need to do a loop until the game is finished
        We will use a list to store [heuristic(child),col] and sort it so the best play for 'O' is first and for 'X' is last
        A* will play as 'O' when vs human, so the lower the score the best (due to our heuristic setup)
        it returns the col from best child based on the turn
        '''
        heuristic = lambda state,col: Game4InLine.heuristic_points(state,col)+ Game4InLine.heuristic_extra(state,col)
        childs=self.childs()
        points_col=[]  #points_col[k][0] = points | points_col[k][1] = col
        #points_given=[] #list to visualise the given points per heuristic and the column played. format-> list of lists = [[h_points,h_extra,col]]
        #DT stuff
        wins = []
        draws = []
        for i in range(len(childs)):
            col=childs[i][1]
            if dt == 0:
                points_col.append([heuristic(state=(childs[i][0]),col=col),col])
                #para visualizar pontuacao de cada heuristica
                #points_given.append([Game4InLine.heuristic_points((childs[i][0]),col),Game4InLine.heuristic_extra((childs[i][0]),col),col+1])

            if dt == 1:
                list_board = [item for sublist in self.board for item in sublist]
                df = pd.DataFrame([list_board])
                df.columns = range(42)
                res = self.decision_tree.predict(df)[0]
                if res == "win":
                    wins.append(col)
                elif res == "draw":
                    draws.append(col)
        
        if dt == 0:
            points_col.sort() #order the list so that first is the lowest points in total and last the max points. !!*this doesn't mean the best play is last*!!
            return (points_col[0][1]) if self.pieces[self.turn] == 'o' else (points_col[-1][1])
        
        elif dt == 1:
            if len(wins) != 0:
                print(f"wining cols: {wins}")
                return random.choice(wins)
            elif len(draws) != 0:
                print(f"drawing cols: {draws}")
                return random.choice(draws)
            else:
                print(f"only loses")
                return random.choice(self.legal_moves())


    def resultado_DT(self):        
        childs=self.childs()
        wins = []
        draws = []
        loses = []
        for i in range(len(childs)):
            col=childs[i][1] + 1
            list_board = [item for sublist in self.board for item in sublist]
            df = pd.DataFrame([list_board])
            df.columns = range(42)
            res = self.decision_tree.predict(df)[0]
            if res == "win":
                wins.append(col)
            elif res == "draw":
                draws.append(col)
            else:
                loses.append(col)

        print(f"Colunas que originam \"wins\": {wins}\nColunas que originam \"draws\": {draws}\n Colunas que originam \"loses\": {loses}")


    def __str__(self): #override the print() method
        return print_board(self.board)




def result(game,col):
    ''' 
    funcao para analisar quando o jogo acabar se foi empate ou qual jogador ganhou
    '''
    res=game.isFinished(col)
    if res:
        if res==2:
            print(f"Draw")
        else:
            print(f"{game.pieces[game.turn-1]} won")
        return True
    return False


def getPoints(x,o):
    '''
    returns the points based on the given heuristic for the project
    '''
    if (x == 4 and o == 0):
        return 512
    if (x == 3 and o == 0):
        return 50
    if (x == 2 and o == 0):
        return 10
    if (x == 1 and o == 0):
        return 1
    if (x == 0 and o == 1):
        return -1
    if (x == 0 and o == 2):
        return -10
    if (x == 0 and o == 3):
        return -50
    if (x == 0 and o == 4):
        return -512
    return 0

def getPoints_extra(game: Game4InLine, x, o):
    '''
    returns the points based on the extra heuristic setup
    '''
    if(x==3 and o==1) and game.pieces[game.turn-1] == 'o':
        return -500
    if(x==2 and o==1) and game.pieces[game.turn-1] == 'o':
        return -100  
    if(x==1 and o==3) and game.pieces[game.turn-1] == 'x':
        return 500
    if(x==1 and o==2) and game.pieces[game.turn-1] == 'x':
        return 100 
    return 0

def print_board(board): #transform the game board from matrix to a visual representation
    board_str="|"
    for k in range(1,len(board[0])+1):
        board_str+=f" {k} |"
    board_str+="\n"
    for i in range(len(board)):
        board_str += "| "
        for j in range(len(board[i])):
            #board_str+=board[i][j]
            peca = "-" if board[i][j] == "b" else board[i][j]
            board_str += f"{peca} | "
        board_str+="\n"
    return board_str