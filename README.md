# PyJigsaw
Puzzle solver written in python. Used to solve BlackHell 1000 piece puzzle I got from Amazon.

# Top level description of the files:
## Modules
### processor.py: 
This performs the image processing. Takes in a jpg file, analyzes it and creates a .npz file that the app works with.
### app.py: 
This is the CLI interface that guides you as you solve the puzzle.
### fitter.py: 
This is an internal module that scores how well two sides of two peices fit together.
### jigsaw.py: 
This is an internal module that records a jigsaw peice.
### legit.py: 
This is a utility tool that gives a quick go/nogo of a given .npz file. Useful for looking at a quick summary of a batch execution of peices analysis.

## Batch file
### all_process.bat: 
Useful to process an entire folder's worth of peices and then give a summary if all peices were analyzed correctly.
### viewall.bat: 
View all NPZ files in a folder.

## Data files
### puzzlemap.py: 
Contains the pattern underneath the puzzle.
### SpecialSolving.py: 
Contains a special path to which the puzzle is to be solved. This will be used only if needed from the app.

