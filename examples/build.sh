#!/usr/bin/env bash
set -e

for F in */; do
    F="${F%/}"
    odin build "$F" -vet -debug -out:../build/"$F".exe
done