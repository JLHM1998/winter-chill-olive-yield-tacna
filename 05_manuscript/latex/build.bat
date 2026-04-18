@echo off
REM Build manuscript + supplementary (single run)
REM Usage:
REM   build          — compila una vez
REM   build watch    — recompila cada vez que guardas un archivo
REM   build clean    — borra archivos auxiliares

cd /d "%~dp0"

if "%1"=="watch" (
    echo Modo watch: recompila automaticamente al guardar...
    latexmk -pvc
) else if "%1"=="clean" (
    latexmk -C
    echo Limpieza completada.
) else (
    latexmk
    echo.
    echo Compilacion completada. Revisa manuscript.pdf y supplementary.pdf
)
