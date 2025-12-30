@echo off
setlocal

set glsl_flags=

for /D %%F in (*) do (
    if exist "%%F\shaders\" (
        for %%S in ("%%F\shaders\*.musl") do (
            ..\build\gpu_compiler.exe "%%S"
        )
    )
)

for /D %%F in (*) do (
    if exist "%%F\shaders\" (
        for %%S in ("%%F\shaders\*.glsl") do (
            glslangvalidator %glsl_flags% -V "%%S" -o "%%F\shaders\%%~nS.spv"
        )
    )
)
