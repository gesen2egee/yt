#!/bin/bash

# 定義顏色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 檢查虛擬環境是否存在
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️ 找不到虛擬環境，正在嘗試自動安裝...${NC}"
    if [ -f "install.sh" ]; then
        chmod +x install.sh
        ./install.sh
    else
        echo -e "${RED}❌ 找不到 install.sh，請確認檔案完整。${NC}"
        exit 1
    fi
fi

# 再次檢查
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ 虛擬環境建立失敗，無法啟動。${NC}"
    exit 1
fi

echo -e "${GREEN}🚀 正在啟動 字幕提取工具...${NC}"
./venv/bin/python yt.py

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 程式執行發生錯誤。${NC}"
    exit 1
fi
