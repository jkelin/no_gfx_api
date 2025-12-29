@echo off
setlocal

set glsl_flags=

for %%f in (*.musl) do (
   ..\build\gpu_compiler.exe "%%f"
)

for %%f in (*.glsl) do (
    glslangvalidator %glsl_flags% -V "%%f" -o "%%~nf.spv"
)

