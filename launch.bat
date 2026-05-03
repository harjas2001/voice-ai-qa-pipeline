@echo off
REM Voice AI QA Pipeline — Windows Launcher
REM Update the path below to match your local or shared folder location

cd /d "C:\path\to\voice-ai-qa-pipeline"

REM Optional: activate a virtual environment
REM call venv\Scripts\activate

REM Start the Flask app in a new terminal window
start cmd /k "python app.py"

REM Allow Flask a few seconds to start before opening the browser
timeout /t 5 /nobreak

REM Open the app in the default browser
start http://127.0.0.1:5000/

exit
