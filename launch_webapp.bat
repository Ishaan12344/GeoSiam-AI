@echo off
echo ============================================================
echo   GeoSiam AI - Disaster Building Damage Assessment Web App
echo ============================================================
echo Starting FastAPI server on http://127.0.0.1:8000 ...
echo Press Ctrl+C to stop the server.
echo.
start "" http://127.0.0.1:8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
