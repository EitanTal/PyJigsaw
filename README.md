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

## Puzzle data
Puzzle data photos are not included due to possible copyright issues.

# Tips
## What did I use when I solved my puzzle
I used a cheapo 640x480 microscope from Amazon to take photos of each peice. I used 7 boxes with 144 compartments each to store the peices after scanning. Many photos needed touchups to clean up microscopic debris that stuck to the peice. I used mspaint to clean up any such dirt. Paid special attention to keeping corners clean and the dot quadrant peices as clean as possible.

## Notation I used
I used this notation. Example:
NPZ file: C:\jigsaw\data\ttt\box1\1a.npz (has a corresponding JPG file in the same folder). Peice ID is therefore ttt/box1/1a
ttt stands for triangles. Others are sss, ddd and ccc (stripes, dots and clubs)

## Things to watch out for:
The processing vulnerable step is the rectangle detection. The processing algorithm sometimes won't find the corners. Most of the time it is due to bad lighting or debris. Cleaning up the troubled image always fixed the problem. 

Sometimes shadows will not get rejected correctly and cause the app to give the correct peice a bad score. That usually wasn't too big of a problem.

When I specified an orientation in the processing step, I made one mistake of scanning one peice in its opposite orientation. Try not to make that mistake. If you think that's the case, run the app with -ort to see if you get better results.

The app displays the board in a mirror image, as I was solving the puzzle with the pattern facing up.

when scanning a batch of peices, scan a reference card to make sure the scale hasn't changed.

## Usage
### Parameters for app:
-new to create a new board

-ort to disregard orientation data

-ext to perform an exhaustive candidate search

### Parameters for processor:
-ort to specify allowed orientations. Do not include if there is no data. e.g. dot quadrant. Example: -ort 10 = [ False True False True ] meaning you're allowed to use this peice rotated 1 or 3 times counterclockwire. This would be a stripes peice because this has 2 possible orientations. -ort 1 = [ True False False False ]. This would be a triangles or clubs peice that is only allowed in its original scanned rotation.

-debugall to view all the steps.

### app keys
0-9 select the suggested peice

b go backwards

? display location

pX,Y set cursor to x,y

m show more candidates

s save

q quit

('ccc/box1/1a', 3) manually specify a particular peice and orientation

### Other app settings:
Change SolvingType in app.py to 'Special' and edit SpecialSolving python file to your liking

# Installation notes
Dependency modules: There are several depencies, mostly numpy and matplotlib. Simply run the scripts and google any errors you come across.

## Important: 
Make sure your python is 64 bit. All puzzle peices are loaded into RAM and the process takes about 4 gigs in ram. This will crash in a 32bit instance.
