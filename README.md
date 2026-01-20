# YouTube/Bilibili 影片字幕提取與 AI 整理工具 (v3.0)
<img width="1176" height="1025" alt="image" src="https://github.com/user-attachments/assets/5d90e87c-1ddc-4641-b463-2f97358c306e" />

這是一個強大且適合初學者的工具，可以幫你從 YouTube、Bilibili 等網站下載影片/音訊，並透過 AI (Google Gemini) 自動產生字幕、摘要，還能幫你整理筆記！

## ✨ 主要功能

* **多平台支援**：支援 YouTube、Bilibili 以及大多數 `yt-dlp` 支援的影音網站。
* **AI 智慧處理**：
  * 利用 Google Gemini AI 進行精準的語音轉文字 (STT) 和摘要生成。
  * 自動過濾空白片段，並進行人聲增強，讓 AI 聽得更清楚。
* **自動化整理**：
  * 自動將內容分類、分組。
  * 提供網頁介面 (Web UI) 讓你輕鬆管理所有影片筆記。
* **分段與優化**：
  * **音訊分段**：自動將長影片切段處理並傳遞上下文，確保連貫。
  * **提詞重複**：支援重複發送提示詞以提升 AI 準確度。
  * **YouTube 合輯**：支援 Playlist 一鍵解析展開。
* **初學者友善**：執行程式時會自動檢查並安裝所需的套件，無需複雜設定。

## 🚀 快速開始

### 1. 環境準備

#### 安裝 Python 3.10+

* **方法一：使用 winget（推薦）**

  ```powershell
  winget install Python.Python.3.12
  ```

* **方法二：官網下載**
  * 前往 [Python 官網](https://www.python.org/downloads/)
  * 下載 Windows 安裝檔
  * 安裝時**務必勾選「Add Python to PATH」**

驗證安裝：

```powershell
python --version
```

#### 安裝 Git

* **方法一：使用 winget（推薦）**

  ```powershell
  winget install Git.Git
  ```

* **方法二：官網下載**
  * 前往 [Git 官網](https://git-scm.com/download/win)
  * 下載並安裝（全部使用預設設定即可）

驗證安裝：

```powershell
git --version
```

#### 安裝 ffmpeg（音訊處理必需）

* **方法一：使用 winget（推薦）**

    ```powershell
    winget install ffmpeg
    ```

* **方法二：手動安裝**
    1. 下載：<https://www.gyan.dev/ffmpeg/builds/> （選擇 ffmpeg-release-essentials.zip）
    2. 解壓到 `C:\ffmpeg`
    3. 將 `C:\ffmpeg\bin` 加入系統環境變數 PATH
    4. 重新啟動終端機

驗證安裝：

```powershell
ffmpeg -version
```

### 2. 下載專案並啟動 (Windows 推薦)

下載此專案後，Windows 使用者可以透過以下腳本進行操作：

* **初次啟動**：點擊 `install.bat`，程式會自動建立虛擬環境 (`venv`) 並安裝所有套件。
* **日常使用**：點擊 `run.bat` 即可啟動程式。

---

### 3. 手動安裝與執行 (進階)

如果你想使用手動指令：

```powershell
git clone <你的專案網址>
cd yt
python yt.py
```

第一次執行時，程式會自動幫你安裝所有需要的 Python 套件 (如 `flask`, `yt-dlp`, `google-genai` 等)，請耐心等待。

### 3. 開始使用

程式啟動後，會自動開啟網頁瀏覽器 (通常是 `http://127.0.0.1:5000`)。
你可以在網頁介面上：

1. **貼上影片網址**：支援單一影片或播放清單。
2. **設定 API Key**：在設定頁面填入你的 Google Gemini API Key (可以免費申請)。
3. **開始任務**：點擊開始，程式就會自動下載、轉錄並生成摘要。

## 📁 檔案結構說明

程式會自動建立一個 `subtitle_data` 資料夾來存放所有資料，包含：

* `subtitles/`: 存放生成的字幕檔。
* `audio_cache/`: 暫存的音訊檔案 (處理完會自動清理)。
* `videos/`: 如果你有勾選下載影片，會存放在這裡。
* `database.json`: 儲存你的所有任務記錄和摘要資料。

**注意**：`subtitle_data` 資料夾、`subtitles` 資料夾以及 `date` 檔案已被設定為忽略 (Git ignored)，不會被上傳到版本控制系統中，確保你的個人資料安全。

## 💡 常見問題

* **API Key 哪裡取得？**
  * 你可以到 [Google AI Studio](https://aistudio.google.com/) 免費申請 Gemini API Key。
* **支援中文影片嗎？**
  * 支援！Gemini 對於中文的理解能力非常好。

## 🛠️ 進階設定

你可以在網頁介面的「設定」中調整：

* **AI 模型**：選擇不同的 Gemini 模型版本。
* **處理模式**：選擇是否要保留靜音片段、加速音訊等。
* **介面顯示**：切換深色/淺色模式等。

---
*由 Antigravity 協助生成與維護*
