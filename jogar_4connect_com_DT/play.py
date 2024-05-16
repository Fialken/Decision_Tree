import time
from Game4InLine import Game4InLine as G4Line
from MCTS import MCTS

BOARD_SIZE_STANDARD = True #make it 'False' if u want to play 4InLine with a board diff from 6x7
TIME_MCTS = 5 #time for search with mcts

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

def main(): 
    ''' 
    funcao para jogar Player vs Player | ou | Player vs IA
    permite configurar uma board diferente do 'normal' definido caso BOARD_SIZE_STANDARD = False sendo a board minima de 5x5
    recebe um input y/n para saber se vai ser jogado contra IA e qual IA (A* ou MCTS)
    no main loop corremos o jogo, onde decidimos qual coluna jogar (column_played) por input (players) ou decisao dos algoritmos (IA)
    loop acaba quando alguem ganhar ou empatar
    '''
    #in case user decides to play with a different board size from 6x7
    game=G4Line(6,7)
    print(f"Board:\n{game}")
    
    #if user want to play vs AI
    Ai_playing = 'y'
    which_AI = 1
    print("")

    #main loop
    while True: 
        #player turn
        print(f"player {game.turn%2 +1} ('{game.pieces[game.turn%2]}') turn")

        #Human play       
        column_played = int(input("Column to place: ")) - 1 # -1 because the columns goes from 0 to 6(or set col size) and user is expected to select a number from 1 to 7 (or set col size)
        while (column_played > game.cols-1 or column_played < 0) or game.placed[column_played] >= game.rows :
            print("Impossible move")
            column_played = int(input("Column to place: ")) - 1

        game.play(column_played)
        print(f"Board:\n{game}")
        
        if result(game,column_played):
            break


        #AI play
        if Ai_playing == 'y':
            if which_AI == 1: #A*
                column_played=game.A_star(1)
                game.play(column_played)
                print(f"A* DT play: {column_played+1}")

            print(f"Board:\n{game}")
            if result(game,column_played):
                break

def main_A_star():
    '''
    this function was created to play A* vs A* without any human interaction
    as we have no variations to play, we will always play the same game/result
    and player 2 (O) is the winner all the time with 30 rounds played
    '''
    game=G4Line(6,7) #board is set to 'normal' size
    #main loop
    while True:     
        #A* with DT
        print(f"A* DT ('{game.pieces[game.turn%2]}') turn") # to know which turn is X(1) or O(2)
        #play 
        column_played=game.A_star(1)
        game.play(column_played)
        print(f"played: {column_played+1}")
        print(f"Board:\n{game}")

        if result(game,column_played):
            break

        
        #A* normal
        print(f"A* heuristic ('{game.pieces[game.turn%2]}') turn") # to know which turn is X(1) or O(2)
        #play 
        column_played=game.A_star(0)
        game.play(column_played)
        print(f"played: {column_played+1}")
        print(f"Board:\n{game}")

        if result(game,column_played):
            break
        
    print(f"game took {game.round} rounds")

def A_star_vs_MCTS(i):
    start_time = time.time() #timer to know how long it takes
    game=G4Line(6,7) #board is set to 'normal' size
    #main loop
    while True:     
        #play 
        if(game.turn %2 == i):
            print(f"A*(DT) ('{game.pieces[game.turn%2]}') turn")
            column_played=game.A_star(1)
            game.play(column_played)
            print(f"A*(DT) play: {column_played+1}")
            print(f"Board:\n{game}")
        else:
            print(f"MCTS ('{game.pieces[game.turn%2]}') turn")
            tree = MCTS(game)
            tree.search(TIME_MCTS) 
            column_played = tree.best_move()
            n_simulations, run_time= tree.statistic()
            game.play(column_played)
            
            print(f"MCTS play: {column_played+1}")
            print(f"Board:\n{game}")
            print(f"Num simulations = {n_simulations} in {run_time:.5f}seg")
        

        if result(game,column_played):
            break

    print(f"game took {(time.time()-start_time):.0f} seg and {game.round} rounds")


if __name__ == "__main__":
    '''
    made so the user can choose what type of game he wants to play/watch
    '''
    qual = int(input(f"Qual modo:\n1: Player vs A*(DT) \n2: A*(DT) vs A*(heuristic)\n3: A*(DT) vs MCTS\nEscolha: "))
    if qual == 1:
        main()
    elif qual == 2:
        main_A_star()
    elif qual == 3:
        A_star_vs_MCTS(0)
