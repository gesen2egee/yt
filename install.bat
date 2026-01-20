@echo off
chcp 65001 > nul
echo 📦 正在建立虛擬環境 (venv)...
python -m venv venv
if %errorlevel% neq 0 (
    echo ❌ 建立虛擬環境失敗，請確認是否已安裝 Python。
    pause
    exit /b
)
echo 🆙 正在升級 pip...
venv\Scripts\python.exe -m pip install --upgrade pip
echo 📥 正在安裝依賴套件...
venv\Scripts\pip.exe install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ 安裝失敗。
    pause
    exit /b
)
echo ✅ 安裝完成！現在可以執行 run.bat 啟動程式。
pause
