#!/bin/bash

# 定義顏色
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "📦 正在檢查 Python 環境..."

# 檢查 python3 是否存在
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ 找不到 python3，請先安裝 Python。${NC}"
    echo "推薦使用: brew install python"
    exit 1
fi

echo "📦 正在建立虛擬環境 (venv)..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 建立虛擬環境失敗。${NC}"
    exit 1
fi

echo "🆙 正在升級 pip..."
./venv/bin/python -m pip install --upgrade pip

echo "📥 正在安裝依賴套件..."
./venv/bin/pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 安裝失敗，請檢查錯誤訊息。${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 安裝完成！現在可以執行 ./run.sh 啟動程式。${NC}"
echo "提示：如果無法執行，請輸入 chmod +x run.sh 給予執行權限。"
