@echo off
setlocal

for /D %%F in (*) do (
    : odin build %%F -vet -debug -out:../build/%%F.exe
    odin build %%F -debug -out:../build/%%F.exe
    if errorlevel 1 (
        exit /b 1
    )
)
