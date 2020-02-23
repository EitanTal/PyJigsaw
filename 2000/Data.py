import sys
from app2000 import Board as Board
from app2000 import populateInventory as populateInventory
from app2000 import moveFwd as moveFwd
from app2000 import getFwdDir as getFwdDir
from app2000 import moveBwd as moveBwd
from app2000 import getCandidates as getCandidates

from puzzlemap import ThePuzzleMap


sx = 50
sy = 40
datadir = r'C:\jigsaw\data\2000\npz'
SolvingType = 'Spiral'

solved = Board(sx,sy)
unsolved = Board(sx,sy)

def niceprint(list, correct_answer):
    if (len(list) == 0): 
        print ('-- No Matches --')
        input("Hit enter to continue").strip()
    for i,elem in enumerate(list):
        if (elem[0][0] == correct_answer[0]):
            print ('*** ', end='')
        else:
            print ('    ', end='')
        mystr = ('{}: {}\t{:.2f}\t({:.2f}\t{:.2f})'.format( i, elem[0], elem[1], elem[2], elem[3]))
        print(mystr)

def solve_simulate():
    pos = moveFwd(None)
    while (pos):
        # find 8-adjacent neighbors: Out of bounds are None, unsolved are ('?', 0)
        nUp = unsolved.at(pos[0],pos[1]-1)
        nDn = unsolved.at(pos[0],pos[1]+1)
        nRt = unsolved.at(pos[0]+1,pos[1])
        nLt = unsolved.at(pos[0]-1,pos[1])
        n7 = unsolved.at(pos[0]-1,pos[1]-1)
        n9 = unsolved.at(pos[0]+1,pos[1]-1)
        n1 = unsolved.at(pos[0]-1,pos[1]+1)
        n3 = unsolved.at(pos[0]+1,pos[1]+1)

        targetq = ThePuzzleMap.splitlines()[pos[1] + 1][pos[0]]
        candidates = getCandidates(nUp, nDn, nRt, nLt,n7,n9,n1,n3, targetq, False)
        correct_answer = solved.at(*pos)
        niceprint(candidates, correct_answer)
        unsolved.update(*pos, correct_answer)
        pos = moveFwd(pos)

        input("Hit enter to continue").strip()

def main():
    solved.load()
    solve_simulate()


main()