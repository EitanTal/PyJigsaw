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
        #input("Hit enter to continue").strip()
        exit(1)
    for i,elem in enumerate(list):
        if (elem[0][0] == correct_answer[0]):
            print ('*** ', end='')
        else:
            print ('    ', end='')
        if (type(elem[1]) is int and elem[1] == 0):
            mystr = ('{}: {}\t-----\t({:.2f}\t{:.2f}) meta:{:.2f}'.format( i, elem[0], elem[2], elem[3], elem[-1]))
        else:
            mystr = ('{}: {}\t{:.2f}\t({:.2f}\t{:.2f}) {} meta:{:.2f}'.format( i, elem[0], elem[1], elem[2], elem[3], elem[4], elem[-1]))
        print(mystr)

def reorder_candidates(candidates):
    # sort by angle
    # remove records > 1100 (nudge >= 15 is discarded via the fitter module)
    results = []
    for c in candidates:
        if True:#c[1] <= 11000:
            # calculate meta-score:
            ang = c[3]
            l = c[2]
            metascore = ang + l/10
            c += [metascore]
            results += [c]

    return results

    # order by meta-score:
    results_ordered = sorted(results,key=lambda x: x[-1])

    return results_ordered

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
        raw_candidates = getCandidates(nUp, nDn, nRt, nLt,n7,n9,n1,n3, targetq, False)
        candidates = reorder_candidates(raw_candidates)
        #candidates = raw_candidates
        correct_answer = solved.at(*pos)
        niceprint(candidates, correct_answer)
        unsolved.update(*pos, correct_answer)
        pos = moveFwd(pos)
        print ('Pos is:', pos)
        #input("Hit enter to continue").strip()

def main():
    solved.load()
    solve_simulate()


main()