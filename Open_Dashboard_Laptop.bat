@echo off
cd /d %~dp0
echo Starting FloodGuard AI dashboard at http://127.0.0.1:8080/
echo Keep this window open while using the dashboard.
start "FloodGuard AI" http://127.0.0.1:8080/FloodGuard_AI_India_Map_ML.html
py -m http.server 8080
pause
