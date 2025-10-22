@echo off
echo Activating main virtual environment...
call "venv_main\Scripts\activate.bat"
echo.
echo Main environment activated!
echo This environment contains: torch, librosa, opencv, transformers, etc.
echo.
echo Repository is now organized:
echo   - Training scripts: scripts\training\
echo   - Evaluation scripts: scripts\evaluation\
echo   - Utility scripts: scripts\utils\
echo   - Cleanup scripts: scripts\cleanup\
echo.
echo To deactivate, run: deactivate
echo.
