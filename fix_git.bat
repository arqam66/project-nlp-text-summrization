@echo off
echo Resetting Git history to remove the large files completely...
rmdir /s /q .git
git init
git add .
git commit -m "Initial commit without large datasets"
git remote add origin https://github.com/arqam66/nlp-project.git
git branch -M main
git push -u origin main --force
echo.
echo Done!
pause
