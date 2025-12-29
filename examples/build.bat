@echo off
setlocal

for /D %%F in (*) do (
    odin build %%F -vet -debug -out:../build/%%F.exe
)
