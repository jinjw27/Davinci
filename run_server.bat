@echo off
echo Starting Da Vinci Code Local Server...
echo ---------------------------------------
echo URL: http://localhost:8000/davinci_game.html
echo ---------------------------------------
:: Open the browser automatically
start http://localhost:8000/davinci_game.html
python -m http.server 8000
pause
