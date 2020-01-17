rem EXAMPLE
rem for /f %%f in ('dir /b c:\jigsaw\data\%1\*.jpg') do echo %1/%%f

rem PROCESS
for /f %%f in ('dir /b c:\jigsaw\data\%1\*.jpg') do processor.py %1/%%f %2 %3

rem VALIDATE
echo off
for /f %%f in ('dir /b c:\jigsaw\data\%1\*.jpg') do legit.py %1/%%f

rem VIEW
rem for /f %%f in ('dir /b c:\jigsaw\data\box9\*.npz') do jigsaw.py box9/%%f

