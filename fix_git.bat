@echo off
echo Committing changes...
git commit --amend --no-edit
echo.
echo Pushing to GitHub...
git push -u origin main
echo.
pause
