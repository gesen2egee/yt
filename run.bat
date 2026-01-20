@echo off
chcp 65001 > nul
if not exist "venv" (
    echo ⚠️ 找不到虛擬環境，正在嘗試自動安裝...
    call install.bat
)
echo 🚀 正在啟動 字幕提取工具...
venv\Scripts\python.exe yt.py
if %errorlevel% neq 0 (
    echo ❌ 程式執行發生錯誤。
)
pause
