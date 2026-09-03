@echo off
echo.
echo  ParseDat_Diary — Chat with Books
echo  RTX 4050 · Local AI · Your Data
echo.


echo Commands:
echo   python main.py ingest          -- Extract PDFs
echo   python main.py chat            -- Chat with books
echo   python main.py server          -- Web API on :8000
echo   python main.py search "query"  -- Quick search
echo   python main.py service install -- Install as Windows service
echo.

python main.py %*
pause
