"""
Code Index:
25   Imports
47   Dependency Installation (install_package)
103  Default Configuration (DEFAULT_CONFIG)
124  Data Models (Job, Summary, UsageRecord)
185  Data Management (DataManager class)
558  Utility Functions (sanitize_filename, timestamps, etc.)
673  Audio Preprocessing (AudioPreprocessor class)
1706 Flask Application Setup
1719 API Routes & Views
2375 HTML Templates
"""

"""
YouTube/Bilibili 字幕提取與整理工具 v3.0
- 改用 Google Gemini AI Studio API（Files API 上傳音檔）
- 支援所有 yt-dlp 可處理的網站
- 人聲加強 → 去靜音 → 加速 音訊前處理
- 分類分組功能
- 可選下載原影片
- API Key 從設定檔讀取
"""

import os
import sys
import json
import re
import uuid
import hashlib
import subprocess
import threading
import queue
import webbrowser
import argparse
import time
import shutil
import concurrent.futures
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Set

# 清理會干擾 yt-dlp 的 proxy 環境變數
PROXY_ENV_KEYS = [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
    "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY",
    "git_http_proxy", "git_https_proxy",
]

def env_without_proxy() -> Dict[str, str]:
    env = os.environ.copy()
    for k in PROXY_ENV_KEYS:
        env.pop(k, None)
    return env

# =============================================================================
# 自動安裝依賴
# =============================================================================

def install_package(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"📦 正在安裝 {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
            return True
        except Exception as e:
            print(f"❌ 安裝 {package_name} 失敗：{e}")
            return False

REQUIRED_PACKAGES = [
    ("flask", "flask"),
    ("requests", "requests"),
    ("yt-dlp", "yt_dlp"),
    ("google-genai", "google.genai"),
]

for pkg, imp in REQUIRED_PACKAGES:
    if not install_package(pkg, imp):
        print(f"請手動安裝: pip install {pkg}")
        sys.exit(1)

from flask import Flask, render_template_string, request, jsonify, send_file
import requests

# =============================================================================
# 配置
# =============================================================================

DATA_DIR = Path("./subtitle_data")
AUDIO_CACHE_DIR = DATA_DIR / "audio_cache"
SUBTITLE_DIR = DATA_DIR / "subtitles"
VIDEO_DIR = DATA_DIR / "videos"
DB_FILE = DATA_DIR / "database.json"

for d in [DATA_DIR, AUDIO_CACHE_DIR, SUBTITLE_DIR, VIDEO_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    
PRICE_CONFIG = {
    # Gemini 價格
    "google/gemini-2.5-flash-preview-09-2025": {"input": 0.3, "output": 2.50, "audio_input": 1.00},
    "google/gemini-3-flash-preview": {"input": 0.5, "output": 3.00, "audio_input": 1.00},
    "google/gemini-3-flash-preview:thinking": {"input": 0.5, "output": 3.00, "audio_input": 1.00},
    "google/gemini-3-pro-preview": {"input": 2.00, "output": 12.00},

    # STT（USD / hour）
    "whisper-large-v3": {"audio_per_hour": 0.111},
    "whisper-large-v3-turbo": {"audio_per_hour": 0.04},
}

USD_TO_TWD = 30

DEFAULT_CONFIG = {
    "gemini_api_key": "",
    "groq_api_key": "",
    "default_llm_model": "google/gemini-3-flash-preview:thinking",
    "default_stt_model": "whisper-large-v3",
    "no_subtitle_action": "llm_direct",  # llm_direct, stt, audio_only
    "audio_format": "m4a",  # m4a, mp3, wav
    "speech_enhance_preset": "strong",  # off, light, medium, strong
    "llm_audio_speed": 1.5,
    "silence_noise_db": -40,
    "silence_min_duration": 1.0,
    "long_video_threshold_minutes": 30,
    "download_video": False,
    "audio_segment_minutes": 10,  # 音訊分段時長（分鐘），0 表示不分段
    "enable_query_repeat": False,  # 啟用提詞重複（提升準確度但會加倍 token）
    "category_groups": {},  # { "group_name": ["cat1", "cat2"], ... }
    "collapsed_groups": [],  # 收縮的分組
}

# =============================================================================
# 資料模型
# =============================================================================

@dataclass
class Job:
    id: str
    url: str
    platform: str = ""
    video_id: str = ""
    status: str = "queued"
    progress: int = 0
    stage: str = "等待中"
    title: str = ""
    channel: str = ""
    uploader: str = ""
    upload_date: str = ""
    duration: int = 0
    subtitle_path: str = ""
    subtitle_content: str = ""
    subtitle_with_time: str = ""
    audio_path: str = ""
    video_path: str = ""
    error_message: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    cancel_requested: bool = False
    deleted: bool = False
    has_original_subtitle: bool = False

@dataclass
class Summary:
    id: str
    job_id: str
    title: str
    content: str
    content_with_time: str = ""
    video_url: str = ""
    video_title: str = ""
    channel: str = ""
    uploader: str = ""
    upload_date: str = ""
    category: str = "未分類"
    model_used: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    pinned: bool = False
    pinned_at: str = ""

@dataclass
class UsageRecord:
    id: str
    timestamp: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    audio_seconds: float = 0.0
    cost_usd: float = 0.0
    cost_twd: float = 0.0
    description: str = ""

# =============================================================================
# 資料管理
# =============================================================================

class DataManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.summaries: Dict[str, Summary] = {}
        self.categories: Dict[str, List[str]] = {"未分類": []}
        self.usage_records: List[UsageRecord] = []
        self.usage_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_audio_seconds": 0,
            "total_cost_usd": 0.0,
            "total_cost_twd": 0.0,
            "by_model": {},
            "stt_usage": {},
        }
        self.config = DEFAULT_CONFIG.copy()
        self._load_data()
        self._lock = threading.Lock()

    def _load_data(self):
        if DB_FILE.exists():
            try:
                with open(DB_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for job_data in data.get("jobs", []):
                    defaults = {
                        "channel": "", "uploader": "", "upload_date": "", "duration": 0,
                        "stage": "等待中", "cancel_requested": False,
                        "deleted": False, "video_path": "", "has_original_subtitle": False,
                        "status": job_data.get("status", "queued") or "queued",
                    }
                    for k, v in defaults.items():
                        if k not in job_data:
                            job_data[k] = v
                    # 移除舊欄位
                    for old_key in ["title_zh", "stt_model", "audio_processed_path", "audio_sped_path", 
                                    "audio_time_map_path", "audio_processing_speed", "audio_duration",
                                    "needs_confirmation"]:
                        job_data.pop(old_key, None)
                    self.jobs[job_data["id"]] = Job(**job_data)

                for summary_data in data.get("summaries", []):
                    for key in ["channel", "uploader", "upload_date", "video_title"]:
                        if key not in summary_data:
                            summary_data[key] = ""
                    self.summaries[summary_data["id"]] = Summary(**summary_data)

                self.categories = data.get("categories", {"未分類": []})
                self.usage_stats = data.get("usage_stats", self.usage_stats)
                if "stt_usage" not in self.usage_stats:
                    self.usage_stats["stt_usage"] = {}
                if "total_cost_twd" not in self.usage_stats:
                    self.usage_stats["total_cost_twd"] = self.usage_stats.get("total_cost", 0) * USD_TO_TWD
                if "total_cost_usd" not in self.usage_stats:
                    self.usage_stats["total_cost_usd"] = self.usage_stats.get("total_cost", 0)

                for rec in data.get("usage_records", []):
                    self.usage_records.append(UsageRecord(**rec))

                loaded_config = data.get("config", {})
                for k, v in DEFAULT_CONFIG.items():
                    self.config[k] = loaded_config.get(k, v)

            except Exception as e:
                print(f"載入資料失敗: {e}")

        if "未分類" not in self.categories:
            self.categories["未分類"] = []
        
        # 從環境變數讀取 API Key（優先）
        env_gemini = os.getenv("GEMINI_API_KEY", "")
        env_groq = os.getenv("GROQ_API_KEY", "")
        if env_gemini:
            self.config["gemini_api_key"] = env_gemini
        if env_groq:
            self.config["groq_api_key"] = env_groq

    def _save_data(self):
        data = {
            "jobs": [asdict(j) for j in self.jobs.values()],
            "summaries": [asdict(s) for s in self.summaries.values()],
            "categories": self.categories,
            "usage_stats": self.usage_stats,
            "usage_records": [asdict(r) for r in self.usage_records],
            "config": self.config,
        }
        tmp = DB_FILE.with_suffix(".json.tmp")
        with self._lock:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(DB_FILE)

    def add_job(self, job: Job):
        self.jobs[job.id] = job
        self._save_data()

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def update_job(self, job: Job):
        existing = self.jobs.get(job.id)
        if existing and getattr(existing, "deleted", False):
            return
        self.jobs[job.id] = job
        self._save_data()

    def soft_delete_job(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job:
            return
        job.deleted = True
        job.cancel_requested = True
        if job.status not in ("completed", "error"):
            job.status = "cancelled"
            job.stage = "已刪除/取消"
        self._save_data()

    def add_summary(self, summary: Summary):
        self.summaries[summary.id] = summary
        if summary.category not in self.categories:
            self.categories[summary.category] = []
        if summary.id not in self.categories[summary.category]:
            self.categories[summary.category].append(summary.id)
        self._save_data()

    def add_usage_record(self, model: str, input_tokens: int, output_tokens: int,
                         audio_seconds: float = 0, description: str = ""):
        base_model = model.replace(":thinking", "")
        price = PRICE_CONFIG.get(base_model, {"input": 0, "output": 0, "audio_input": 0})

        if audio_seconds > 0:
            audio_per_hour = price.get("audio_per_hour", 0)
            if audio_per_hour > 0:
                cost_usd = (audio_seconds / 3600) * audio_per_hour
            else:
                audio_tokens = int(audio_seconds * 25)
                cost_usd = (audio_tokens * price.get("audio_input", 1.0)) / 1_000_000
        else:
            cost_usd = (input_tokens * price.get("input", 0) + output_tokens * price.get("output", 0)) / 1_000_000

        cost_twd = cost_usd * USD_TO_TWD

        record = UsageRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            audio_seconds=audio_seconds,
            cost_usd=cost_usd,
            cost_twd=cost_twd,
            description=description
        )
        self.usage_records.insert(0, record)

        self.usage_stats["total_input_tokens"] += input_tokens
        self.usage_stats["total_output_tokens"] += output_tokens
        self.usage_stats["total_audio_seconds"] += audio_seconds
        self.usage_stats["total_cost_usd"] = self.usage_stats.get("total_cost_usd", 0) + cost_usd
        self.usage_stats["total_cost_twd"] = self.usage_stats.get("total_cost_twd", 0) + cost_twd

        if model not in self.usage_stats["by_model"]:
            self.usage_stats["by_model"][model] = {"input": 0, "output": 0, "cost_usd": 0.0, "cost_twd": 0.0}
        self.usage_stats["by_model"][model]["input"] += input_tokens
        self.usage_stats["by_model"][model]["output"] += output_tokens
        self.usage_stats["by_model"][model]["cost_usd"] = self.usage_stats["by_model"][model].get("cost_usd", 0) + cost_usd
        self.usage_stats["by_model"][model]["cost_twd"] = self.usage_stats["by_model"][model].get("cost_twd", 0) + cost_twd

        self._save_data()
        return record

    def add_stt_usage(self, model: str, duration_seconds: float):
        price = PRICE_CONFIG.get(model, {}).get("audio_per_hour", 0)
        cost_usd = (duration_seconds / 3600) * price
        cost_twd = cost_usd * USD_TO_TWD

        self.usage_stats["total_audio_seconds"] += duration_seconds
        self.usage_stats["total_cost_usd"] = self.usage_stats.get("total_cost_usd", 0) + cost_usd
        self.usage_stats["total_cost_twd"] = self.usage_stats.get("total_cost_twd", 0) + cost_twd

        if model not in self.usage_stats["stt_usage"]:
            self.usage_stats["stt_usage"][model] = {"seconds": 0, "cost_usd": 0.0, "cost_twd": 0.0}
        self.usage_stats["stt_usage"][model]["seconds"] += duration_seconds
        self.usage_stats["stt_usage"][model]["cost_usd"] = self.usage_stats["stt_usage"][model].get("cost_usd", 0) + cost_usd
        self.usage_stats["stt_usage"][model]["cost_twd"] = self.usage_stats["stt_usage"][model].get("cost_twd", 0) + cost_twd

        record = UsageRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            model=model,
            audio_seconds=duration_seconds,
            cost_usd=cost_usd,
            cost_twd=cost_twd,
            description=f"STT 轉錄 {duration_seconds:.1f}秒"
        )
        self.usage_records.insert(0, record)
        self._save_data()

    def clear_usage_records(self):
        self.usage_records = []
        self.usage_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_audio_seconds": 0,
            "total_cost_usd": 0.0,
            "total_cost_twd": 0.0,
            "by_model": {},
            "stt_usage": {},
        }
        self._save_data()

    def delete_summary(self, summary_id: str):
        if summary_id not in self.summaries:
            return False
        summary = self.summaries[summary_id]
        if summary.category in self.categories and summary_id in self.categories[summary.category]:
            self.categories[summary.category].remove(summary_id)
        del self.summaries[summary_id]
        self._save_data()
        return True

    def delete_summaries_batch(self, summary_ids: List[str]):
        for sid in summary_ids:
            if sid in self.summaries:
                summary = self.summaries[sid]
                if summary.category in self.categories and sid in self.categories[summary.category]:
                    self.categories[summary.category].remove(sid)
                del self.summaries[sid]
        self._save_data()

    def move_summaries_batch(self, summary_ids: List[str], new_category: str):
        if new_category not in self.categories:
            self.categories[new_category] = []
        for sid in summary_ids:
            if sid in self.summaries:
                summary = self.summaries[sid]
                old_cat = summary.category
                if old_cat in self.categories and sid in self.categories[old_cat]:
                    self.categories[old_cat].remove(sid)
                if sid not in self.categories[new_category]:
                    self.categories[new_category].append(sid)
                summary.category = new_category
        self._save_data()

    def toggle_pin_summary(self, summary_id: str):
        if summary_id not in self.summaries:
            return False
        summary = self.summaries[summary_id]
        summary.pinned = not summary.pinned
        summary.pinned_at = datetime.now().isoformat() if summary.pinned else ""
        self._save_data()
        return True

    def update_summary_title(self, summary_id: str, new_title: str):
        if summary_id not in self.summaries:
            return False
        self.summaries[summary_id].title = new_title
        self._save_data()
        return True

    def update_summary_content(self, summary_id: str, new_content: str):
        if summary_id not in self.summaries:
            return False
        self.summaries[summary_id].content = new_content
        self.summaries[summary_id].content_with_time = new_content
        self._save_data()
        return True

    def move_summary(self, summary_id: str, new_category: str):
        if summary_id not in self.summaries:
            return False
        summary = self.summaries[summary_id]
        old_category = summary.category
        if old_category in self.categories and summary_id in self.categories[old_category]:
            self.categories[old_category].remove(summary_id)
        if new_category not in self.categories:
            self.categories[new_category] = []
        if summary_id not in self.categories[new_category]:
            self.categories[new_category].append(summary_id)
        summary.category = new_category
        self._save_data()
        return True

    def add_category(self, name: str):
        if name in self.categories:
            return False
        self.categories[name] = []
        self._save_data()
        return True

    def rename_category(self, old_name: str, new_name: str):
        if old_name not in self.categories or old_name == "未分類":
            return False
        if new_name in self.categories:
            return False
        self.categories[new_name] = self.categories.pop(old_name)
        for summary_id in self.categories[new_name]:
            if summary_id in self.summaries:
                self.summaries[summary_id].category = new_name
        # 更新分組中的分類名
        for group_name, cats in self.config.get("category_groups", {}).items():
            if old_name in cats:
                cats[cats.index(old_name)] = new_name
        self._save_data()
        return True

    def delete_category(self, category_name: str):
        if category_name not in self.categories or category_name == "未分類":
            return False
        for summary_id in self.categories[category_name]:
            self.categories["未分類"].append(summary_id)
            if summary_id in self.summaries:
                self.summaries[summary_id].category = "未分類"
        del self.categories[category_name]
        # 從分組中移除
        for group_name, cats in self.config.get("category_groups", {}).items():
            if category_name in cats:
                cats.remove(category_name)
        self._save_data()
        return True

    def add_category_group(self, group_name: str):
        if "category_groups" not in self.config:
            self.config["category_groups"] = {}
        if group_name in self.config["category_groups"]:
            return False
        self.config["category_groups"][group_name] = []
        self._save_data()
        return True

    def delete_category_group(self, group_name: str):
        if group_name in self.config.get("category_groups", {}):
            del self.config["category_groups"][group_name]
            self._save_data()
            return True
        return False

    def add_category_to_group(self, category_name: str, group_name: str):
        if group_name not in self.config.get("category_groups", {}):
            return False
        if category_name not in self.config["category_groups"][group_name]:
            # 先從其他分組移除
            for gn, cats in self.config["category_groups"].items():
                if category_name in cats:
                    cats.remove(category_name)
            self.config["category_groups"][group_name].append(category_name)
            self._save_data()
        return True

    def remove_category_from_group(self, category_name: str, group_name: str):
        if group_name in self.config.get("category_groups", {}):
            if category_name in self.config["category_groups"][group_name]:
                self.config["category_groups"][group_name].remove(category_name)
                self._save_data()
                return True
        return False

    def toggle_group_collapse(self, group_name: str):
        if "collapsed_groups" not in self.config:
            self.config["collapsed_groups"] = []
        if group_name in self.config["collapsed_groups"]:
            self.config["collapsed_groups"].remove(group_name)
        else:
            self.config["collapsed_groups"].append(group_name)
        self._save_data()

data_manager = DataManager()

# =============================================================================
# 工具函數
# =============================================================================

def sanitize_filename(name: str, max_length: int = 80) -> str:
    """清理檔名，移除不合法字元"""
    # 移除 Windows 不允許的字元
    invalid_chars = r'<>:"/\|?*'
    for c in invalid_chars:
        name = name.replace(c, '_')
    # 移除控制字元
    name = re.sub(r'[\x00-\x1f\x7f]', '', name)
    # 移除前後空白和點
    name = name.strip(' .')
    # 限制長度
    if len(name) > max_length:
        name = name[:max_length]
    return name or "untitled"

def extract_video_info(url: str) -> Tuple[str, str]:
    """從 URL 提取平台和影片 ID"""
    yt_patterns = [r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})']
    for pattern in yt_patterns:
        match = re.search(pattern, url)
        if match:
            return ("youtube", match.group(1))
    bili_patterns = [
        r'bilibili\.com/video/(BV[a-zA-Z0-9]+)',
        r'bilibili\.com/video/(av\d+)',
        r'b23\.tv/([a-zA-Z0-9]+)',
    ]
    for pattern in bili_patterns:
        match = re.search(pattern, url)
        if match:
            return ("bilibili", match.group(1))
    # 其他網站：用 URL hash
    return ("other", hashlib.md5(url.encode()).hexdigest()[:12])

def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def generate_video_url_with_time(platform: str, video_id: str, seconds: float, original_url: str = "") -> str:
    t = int(max(0, seconds))
    if platform == "youtube":
        return f"https://www.youtube.com/watch?v={video_id}&t={t}s"
    elif platform == "bilibili":
        return f"https://www.bilibili.com/video/{video_id}?t={t}"
    elif original_url:
        # 其他網站嘗試加 t 參數
        if '?' in original_url:
            return f"{original_url}&t={t}"
        return f"{original_url}?t={t}"
    return ""

def format_upload_date(date_str: str) -> str:
    if date_str and len(date_str) == 8:
        return f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"
    return date_str or ""

def time_str_to_seconds(ts: str) -> int:
    parts = ts.split(":")
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    return 0

def seconds_to_time_str(sec: int) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def linkify_timestamps_in_text(text: str, platform: str, video_id: str, original_url: str = "") -> str:
    if not text:
        return text
    pattern = re.compile(r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]')
    def repl(m):
        ts = m.group(1)
        sec = time_str_to_seconds(ts)
        url = generate_video_url_with_time(platform, video_id, sec, original_url)
        if not url:
            return m.group(0)
        return f'[{ts}]({url})'
    return pattern.sub(repl, text)

def get_audio_duration_seconds(audio_path: Path) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        if r.returncode == 0:
            v = (r.stdout or "").strip()
            return float(v) if v else 0.0
    except Exception:
        pass
    return 0.0

def split_audio_by_duration(audio_path: Path, segment_minutes: int, output_dir: Path) -> List[Path]:
    """
    將音訊按時長切割成多段
    
    Args:
        audio_path: 原始音訊檔案路徑
        segment_minutes: 每段的分鐘數
        output_dir: 輸出目錄
    
    Returns:
        分段檔案路徑列表，如果不需要切割則返回包含原始檔案的單元素列表
    """
    if segment_minutes <= 0:
        return [audio_path]
    
    total_seconds = get_audio_duration_seconds(audio_path)
    segment_seconds = segment_minutes * 60
    
    # 如果音訊時長小於分段時長，不需要切割
    if total_seconds <= segment_seconds:
        return [audio_path]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    segment_count = int((total_seconds // segment_seconds) + (1 if total_seconds % segment_seconds > 0 else 0))
    
    base_name = audio_path.stem
    ext = audio_path.suffix
    
    for i in range(segment_count):
        start_time = i * segment_seconds
        segment_path = output_dir / f"{base_name}_seg{i+1:03d}{ext}"
        
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-i", str(audio_path),
            "-ss", str(start_time),
            "-t", str(segment_seconds),
            "-c", "copy",
            str(segment_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
            if segment_path.exists():
                segments.append(segment_path)
        except Exception as e:
            print(f"切割音訊失敗 (segment {i+1}): {e}")
            # 如果切割失敗，返回原始檔案
            return [audio_path]
    
    return segments if segments else [audio_path]


# =============================================================================
# 音訊前處理
# =============================================================================

class AudioPreprocessor:
    SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
    SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")

    @staticmethod
    def speech_enhance(audio_path: Path, out_path: Path, preset: str = "strong") -> Path:
        """簡易人聲加強"""
        preset = (preset or "off").lower()
        if preset == "off":
            return audio_path

        # 轉成 mono + 帶通濾波
        base = "pan=mono|c0=0.5*c0+0.5*c1,highpass=f=120,lowpass=f=3800"

        if preset == "light":
            af = base
        elif preset == "medium":
            af = base + ",dynaudnorm=f=200:g=5"
        else:  # strong
            af = base + ",acompressor=threshold=-18dB:ratio=4:attack=20:release=200,dynaudnorm=f=200:g=7"

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-i", str(audio_path),
            "-vn", "-filter:a", af,
            "-ac", "1", "-c:a", "aac", "-b:a", "32k",
            str(out_path)
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return out_path if out_path.exists() else audio_path

    @staticmethod
    def detect_silence(audio_path: Path, noise_db: float = -40, min_duration: float = 1.0) -> List[Tuple[float, float]]:
        noise = f"{noise_db}dB"
        d = float(min_duration)
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats",
            "-i", str(audio_path),
            "-af", f"silencedetect=noise={noise}:d={d}",
            "-f", "null", "-"
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        stderr = r.stderr or ""

        intervals: List[Tuple[float, float]] = []
        pending_start: Optional[float] = None

        for line in stderr.splitlines():
            m1 = AudioPreprocessor.SILENCE_START_RE.search(line)
            if m1:
                pending_start = float(m1.group(1))
                continue
            m2 = AudioPreprocessor.SILENCE_END_RE.search(line)
            if m2 and pending_start is not None:
                end = float(m2.group(1))
                start = float(pending_start)
                if end > start:
                    intervals.append((start, end))
                pending_start = None

        return AudioPreprocessor._merge_intervals(intervals)

    @staticmethod
    def _merge_intervals(intervals: List[Tuple[float, float]], eps: float = 1e-3) -> List[Tuple[float, float]]:
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            ls, le = merged[-1]
            if s <= le + eps:
                merged[-1] = (ls, max(le, e))
            else:
                merged.append((s, e))
        return merged

    @staticmethod
    def build_keep_segments(silences: List[Tuple[float, float]], total_duration: float) -> List[Tuple[float, float]]:
        if total_duration <= 0:
            return [(0.0, 0.0)]
        if not silences:
            return [(0.0, total_duration)]

        keep: List[Tuple[float, float]] = []
        cur = 0.0
        for s, e in silences:
            s = max(0.0, s)
            e = min(total_duration, e)
            if s > cur:
                keep.append((cur, s))
            cur = max(cur, e)
        if cur < total_duration:
            keep.append((cur, total_duration))

        keep = [(s, e) for (s, e) in keep if (e - s) >= 0.05]
        if not keep:
            keep = [(0.0, total_duration)]
        return keep

    @staticmethod
    def build_time_map(keep_segments: List[Tuple[float, float]], total_duration: float) -> Dict[str, Any]:
        out_t = 0.0
        segs = []
        for (os, oe) in keep_segments:
            dur = max(0.0, oe - os)
            segs.append({
                "orig_start": os,
                "orig_end": oe,
                "out_start": out_t,
                "out_end": out_t + dur
            })
            out_t += dur

        return {
            "version": 1,
            "orig_duration": float(total_duration),
            "out_duration": float(out_t),
            "segments": segs,
        }

    @staticmethod
    def map_out_to_orig(t_out: float, time_map: Dict[str, Any]) -> float:
        t_out = float(max(0.0, t_out))
        segs = time_map.get("segments") or []
        if not segs:
            return t_out
        if t_out <= 0:
            return float(segs[0].get("orig_start", 0.0))
        if t_out >= float(time_map.get("out_duration", 0.0)):
            last = segs[-1]
            return float(last.get("orig_end", 0.0))

        for seg in segs:
            a = float(seg["out_start"])
            b = float(seg["out_end"])
            if a <= t_out <= b:
                return float(seg["orig_start"]) + (t_out - a)
        return t_out

    @staticmethod
    def chain_atempo(speed: float) -> str:
        speed = float(speed)
        if abs(speed - 1.0) < 1e-6:
            return "atempo=1.0"
        parts = []
        remain = speed
        while remain > 2.0 + 1e-6:
            parts.append("2.0")
            remain /= 2.0
        while remain < 0.5 - 1e-6:
            parts.append("0.5")
            remain /= 0.5
        parts.append(f"{remain:.6f}".rstrip("0").rstrip("."))
        return ",".join([f"atempo={p}" for p in parts])

    @staticmethod
    def render_nosilence(audio_path: Path, keep_segments: List[Tuple[float, float]], out_path: Path) -> Path:
        if len(keep_segments) == 1:
            os0, oe0 = keep_segments[0]
            if os0 <= 0.01:
                shutil.copy(audio_path, out_path)
                return out_path if out_path.exists() else audio_path

        parts = []
        concat_inputs = []
        for i, (s, e) in enumerate(keep_segments):
            parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]")
            concat_inputs.append(f"[a{i}]")
        n = len(keep_segments)
        fc = ";".join(parts) + f";{''.join(concat_inputs)}concat=n={n}:v=0:a=1[aout]"

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-i", str(audio_path),
            "-filter_complex", fc,
            "-map", "[aout]",
            "-ac", "1", "-c:a", "aac", "-b:a", "32k",
            str(out_path)
        ]
        subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return out_path if out_path.exists() else audio_path

    @staticmethod
    def make_sped(nosilence_path: Path, speed: float, out_path: Path) -> Path:
        speed = float(speed or 1.0)
        if abs(speed - 1.0) < 1e-6:
            return nosilence_path

        atempo = AudioPreprocessor.chain_atempo(speed)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-i", str(nosilence_path),
            "-vn", "-filter:a", atempo,
            "-ac", "1", "-c:a", "aac", "-b:a", "32k",
            str(out_path)
        ]
        subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return out_path if out_path.exists() else nosilence_path

    @staticmethod
    def preprocess(audio_path: Path, cache_prefix: str, 
                   speech_enhance_preset: str = "strong",
                   noise_db: float = -40,
                   min_duration: float = 1.0, 
                   speed: float = 1.5) -> Tuple[Path, Dict[str, Any]]:
        """
        完整前處理流程：人聲加強 → 去靜音 → 加速
        返回最終處理後的音檔路徑和時間映射
        """
        # Step 1: 人聲加強
        enhanced_path = AUDIO_CACHE_DIR / f"{cache_prefix}_enhanced.m4a"
        if speech_enhance_preset != "off":
            AudioPreprocessor.speech_enhance(audio_path, enhanced_path, speech_enhance_preset)
            if not enhanced_path.exists():
                enhanced_path = audio_path
        else:
            enhanced_path = audio_path

        # Step 2: 去靜音
        total = get_audio_duration_seconds(enhanced_path)
        silences = AudioPreprocessor.detect_silence(enhanced_path, noise_db=noise_db, min_duration=min_duration)
        keep = AudioPreprocessor.build_keep_segments(silences, total_duration=total)
        time_map = AudioPreprocessor.build_time_map(keep, total_duration=total)
        time_map["silences"] = [{"start": s, "end": e} for (s, e) in silences]
        time_map["keep_segments"] = [{"start": s, "end": e} for (s, e) in keep]
        time_map["speed"] = float(speed)

        nosilence_path = AUDIO_CACHE_DIR / f"{cache_prefix}_nosilence.m4a"
        AudioPreprocessor.render_nosilence(enhanced_path, keep, nosilence_path)
        if not nosilence_path.exists():
            nosilence_path = enhanced_path

        # Step 3: 加速
        if abs(float(speed) - 1.0) < 1e-6:
            final_path = nosilence_path
        else:
            sped_path = AUDIO_CACHE_DIR / f"{cache_prefix}_sped.m4a"
            final_path = AudioPreprocessor.make_sped(nosilence_path, speed, sped_path)

        return final_path, time_map

    @staticmethod
    def cleanup_cache(cache_prefix: str):
        """清理暫存檔案"""
        patterns = [
            f"{cache_prefix}_enhanced.m4a",
            f"{cache_prefix}_nosilence.m4a",
            f"{cache_prefix}_sped.m4a",
        ]
        for pattern in patterns:
            p = AUDIO_CACHE_DIR / pattern
            if p.exists():
                try:
                    p.unlink()
                except:
                    pass

def remap_timestamp_seconds(t_processed: float, speed: float, time_map: Dict[str, Any]) -> float:
    """將處理後的時間戳轉換回原始時間"""
    speed = float(speed or 1.0)
    t_processed = float(max(0.0, t_processed))
    t_nosilence = t_processed * speed
    return AudioPreprocessor.map_out_to_orig(t_nosilence, time_map)

def remap_timestamps_in_text(text: str, speed: float, time_map: Dict[str, Any]) -> str:
    """將文本中的時間戳轉換回原始時間"""
    if not time_map or not speed or abs(float(speed) - 1.0) < 1e-6:
        return text
    pattern = re.compile(r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]')
    def repl(m):
        ts = m.group(1)
        sec_processed = time_str_to_seconds(ts)
        sec_orig = remap_timestamp_seconds(sec_processed, speed, time_map)
        return f"[{seconds_to_time_str(int(round(sec_orig)))}]"
    return pattern.sub(repl, text)

# =============================================================================
# yt-dlp 處理器
# =============================================================================

class YTDLPProcessor:
    LANG_PRIORITY = ["zh-TW", "zh-Hant", "zh", "zh-CN", "zh-Hans", "en", "ja", "ko"]

    def __init__(self):
        self.last_audio_error: str = ""
        self.js_runtime = self._detect_js_runtime()

    @staticmethod
    def _detect_js_runtime() -> str:
        for runtime in ("deno", "node", "bun"):
            if shutil.which(runtime):
                return runtime
        return ""

    @staticmethod
    def _is_youtube_url(url: str) -> bool:
        return bool(re.search(r"(youtube\.com|youtu\.be)", url or "", re.IGNORECASE))

    def _base_cmd(self, url: str) -> List[str]:
        cmd = ["yt-dlp"]
        if self._is_youtube_url(url):
            cmd.extend([
                "--cookies-from-browser", "chrome",
                "--extractor-args", "youtube:player_client=web,default,-tv",
            ])
            if self.js_runtime:
                cmd.extend([
                    "--js-runtimes", self.js_runtime,
                    "--remote-components", "ejs:github",
                ])
        return cmd

    def get_sub_lang_candidates(self, info: dict, max_extra: int = 8) -> List[str]:
        """
        回傳嘗試下載的字幕語言順序：
        1) 先跑 LANG_PRIORITY（只取 info 裡真的存在的）
        2) 再補上一些其他可用語言（避免只剩西文/法文時抓不到）
        """
        subs = info.get("subtitles") or {}
        auto = info.get("automatic_captions") or {}

        preferred = [l for l in self.LANG_PRIORITY if (l in subs and subs.get(l)) or (l in auto and auto.get(l))]

        # 其他語言（先 subs 再 auto）
        others = []
        for l in list(subs.keys()) + list(auto.keys()):
            if l not in preferred:
                others.append(l)

        if max_extra and max_extra > 0:
            others = others[:max_extra]

        return preferred + others

    def get_video_info(self, url: str) -> dict:
        cmd = self._base_cmd(url) + ["--dump-json", "--no-download", "--no-playlist", "--no-warnings", url]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=60, env=env_without_proxy()
            )
            if result.returncode == 0 and result.stdout.strip():
                first_line = result.stdout.strip().split('\n')[0]
                return json.loads(first_line)
        except Exception as e:
            print(f"取得影片資訊失敗: {e}")
        return {}

    def get_playlist_urls(self, url: str) -> List[str]:
        """如果是合輯，展開所有影片網址"""
        cmd = self._base_cmd(url) + ["--flat-playlist", "--print", "%(webpage_url)s", "--no-warnings", url]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=60, env=env_without_proxy()
            )
            if result.returncode == 0:
                urls = []
                for line in result.stdout.splitlines():
                    line = (line or "").strip()
                    if not line or line in ("NA", "None"):
                        continue
                    if line.startswith("http://") or line.startswith("https://"):
                        urls.append(line)
                if urls:
                    return urls
        except Exception as e:
            print(f"解析合輯失敗: {e}")
        return []

    def has_any_subtitles(self, info: dict) -> bool:
        subs = info.get("subtitles") or {}
        auto = info.get("automatic_captions") or {}
        return bool(subs) or bool(auto)

    def download_subtitle(self, url: str, output_dir: Path, title: str, video_id: str, info: dict = None) -> Optional[Tuple[Path, str]]:
        if info is not None:
            subs = info.get("subtitles") or {}
            auto = info.get("automatic_captions") or {}
            if not subs and not auto:
                print("ℹ️ 此影片無可用字幕")
                return None
            langs = self.get_sub_lang_candidates(info, max_extra=8)
            if not langs:
                print("ℹ️ 無符合條件的字幕語言")
                return None
        else:
            langs = self.LANG_PRIORITY

        safe_title = sanitize_filename(title)
        output_name = f"{safe_title}_{video_id}"
        output_template = str(output_dir / output_name)

        # 清理舊檔案
        for f in output_dir.glob(f"{output_name}*"):
            try:
                f.unlink()
            except:
                pass

        # 一次下載：人工 + 自動；多語言一次丟
        langs_csv = ",".join(langs)
        cmd = self._base_cmd(url) + [
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs", langs_csv,
            "--sub-format", "vtt/best",      # 建議先拿 vtt（最常見）
            "--skip-download",
            "--no-playlist", "--no-warnings",
            "-o", output_template,
            url
        ]

        try:
            print(f"📝 嘗試下載字幕（語言優先順序: {', '.join(langs[:5])}...）")
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=120, env=env_without_proxy()
            )
            if result.returncode != 0:
                print(f"⚠️ yt-dlp 字幕下載返回碼: {result.returncode}")
        except subprocess.TimeoutExpired:
            print("❌ 字幕下載超時（120秒）")
            return None
        except Exception as e:
            print(f"❌ 字幕下載失敗: {type(e).__name__}: {e}")
            return None

        # 找所有產出的字幕
        files = list(output_dir.glob(f"{output_name}*.vtt")) + list(output_dir.glob(f"{output_name}*.srt"))
        if not files:
            print("ℹ️ 未找到下載的字幕檔案")
            return None

        print(f"✅ 找到 {len(files)} 個字幕檔案")

        # 用優先順序挑：檔名通常會包含 .<lang>.vtt / .<lang>.srt
        # 這邊用「包含 .{lang}. 」做簡單匹配（比硬切字串穩一點）
        for lang in langs:
            for f in files:
                name = f.name
                if f".{lang}." in name:
                    print(f"✅ 選擇字幕: {f.name} (語言: {lang})")
                    return (f, lang)

        # 如果檔名沒帶 lang（少數情況），就隨便回傳一個
        print(f"✅ 選擇字幕: {files[0].name} (語言: unknown)")
        return (files[0], "unknown")


    def download_audio(self, url: str, output_dir: Path, title: str, video_id: str, 
                       audio_format: str = "m4a") -> Optional[Path]:
        self.last_audio_error = ""
        safe_title = sanitize_filename(title)
        output_name = f"{safe_title}_{video_id}"
        output_path = output_dir / f"{output_name}.{audio_format}"
        
        # 如果檔案已存在，直接返回
        if output_path.exists():
            print(f"✅ 音檔已存在: {output_path.name}")
            return output_path

        format_map = {
            "m4a": ("m4a", "aac", "64k"),
            "mp3": ("mp3", "libmp3lame", "128k"),
            "wav": ("wav", "pcm_s16le", None),
        }
        ext, codec, bitrate = format_map.get(audio_format, ("m4a", "aac", "64k"))

        pp_args = f"-ac 1 -c:a {codec}"
        if bitrate:
            pp_args += f" -b:a {bitrate}"

        cmd = self._base_cmd(url) + [
            "-x", "--audio-format", ext,
            "--no-playlist", "--no-warnings",
            "--postprocessor-args", f"ffmpeg:{pp_args}",
            "-o", str(output_dir / f"{output_name}.%(ext)s"), url
        ]
        
        try:
            print(f"📥 開始下載音軌: {title[:50]}...")
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=600, env=env_without_proxy()
            )
            
            # 記錄輸出（用於除錯）
            had_ytdlp_error = result.returncode != 0
            if result.returncode != 0:
                print(f"⚠️ yt-dlp 返回碼: {result.returncode}")
                detail = (result.stderr or result.stdout or "").strip()
                if detail:
                    self.last_audio_error = detail[:800]
                if result.stderr:
                    # 只顯示錯誤訊息的前500字元
                    stderr_preview = result.stderr[:500]
                    print(f"錯誤訊息: {stderr_preview}")
            
            # 檢查目標檔案是否存在
            if output_path.exists():
                print(f"✅ 下載成功: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
                return output_path
            
            # 嘗試找其他可能的音訊檔案（包括可能的格式變異）
            print(f"🔍 搜尋下載的音訊檔案： {output_name}.*")
            found_files = []
            for f in output_dir.glob(f"{output_name}.*"):
                if f.suffix.lower() in [".m4a", ".mp3", ".wav", ".mp4", ".webm", ".opus", ".aac"]:
                    found_files.append(f)
                    print(f"   找到: {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
            
            if found_files:
                # 優先選擇目標格式，否則選第一個
                target_file = found_files[0]
                for f in found_files:
                    if f.suffix.lower() == f".{ext}":
                        target_file = f
                        break
                
                # 嘗試重命名為目標格式
                try:
                    if target_file != output_path:
                        target_file.rename(output_path)
                        print(f"✅ 重命名為: {output_path.name}")
                        return output_path
                    else:
                        return target_file
                except Exception as rename_error:
                    print(f"⚠️ 無法重命名檔案: {rename_error}，使用原檔案")
                    return target_file
            else:
                print(f"❌ 在 {output_dir} 中找不到任何符合的音訊檔案")
                if not had_ytdlp_error:
                    self.last_audio_error = f"yt-dlp 完成但找不到輸出檔: {output_name}.*"
                
        except subprocess.TimeoutExpired:
            print(f"❌ 下載超時（600秒）")
            self.last_audio_error = "yt-dlp 下載超時（600秒）"
        except Exception as e:
            print(f"❌ 下載音軌失敗: {type(e).__name__}: {e}")
            self.last_audio_error = f"{type(e).__name__}: {e}"
            import traceback
            traceback.print_exc()
        
        return None

    def download_video(self, url: str, output_dir: Path, title: str, video_id: str) -> Optional[Path]:
        safe_title = sanitize_filename(title)
        output_name = f"{safe_title}_{video_id}"
        output_template = str(output_dir / f"{output_name}.%(ext)s")

        cmd = self._base_cmd(url) + [
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--no-playlist", "--no-warnings",
            "-o", output_template, url
        ]
        try:
            subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=1800, env=env_without_proxy()
            )
            for f in output_dir.glob(f"{output_name}.*"):
                if f.suffix.lower() in [".mp4", ".mkv", ".webm"]:
                    return f
        except Exception as e:
            print(f"下載影片失敗: {e}")
        return None

# =============================================================================
# 字幕解析器
# =============================================================================

class SubtitleParser:
    @staticmethod
    def parse_time(time_str: str) -> float:
        time_str = time_str.strip().replace(',', '.')
        parts = time_str.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return 0

    @staticmethod
    def parse_srt(content: str) -> List[dict]:
        segments = []
        blocks = re.split(r'\n\s*\n', content.strip())
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 2:
                continue
            time_line = None
            text_lines = []
            for i, line in enumerate(lines):
                if '-->' in line:
                    time_line = line
                    text_lines = lines[i+1:]
                    break
            if not time_line:
                continue
            time_match = re.match(
                r'(\d{1,2}:\d{2}(?::\d{2})?[,.]\d+)\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?[,.]\d+)',
                time_line
            )
            if not time_match:
                continue
            start_time = SubtitleParser.parse_time(time_match.group(1))
            end_time = SubtitleParser.parse_time(time_match.group(2))
            text = ' '.join(text_lines).strip()
            text = re.sub(r'<[^>]+>', '', text)
            if text:
                segments.append({'start': start_time, 'end': end_time, 'text': text})
        return segments

    @staticmethod
    def parse_vtt(content: str) -> List[dict]:
        content = re.sub(r'^WEBVTT.*?\n\n', '', content, flags=re.DOTALL)
        return SubtitleParser.parse_srt(content)

    @staticmethod
    def segments_to_text(segments: List[dict]) -> str:
        seen = set()
        lines = []
        for seg in segments:
            text = seg['text'].strip()
            if text and text not in seen:
                seen.add(text)
                lines.append(text)
        return '\n'.join(lines)

    @staticmethod
    def segments_to_timestamped_text(segments: List[dict], platform: str = "", video_id: str = "", original_url: str = "") -> str:
        lines = []
        for seg in segments:
            timestamp = format_timestamp(seg['start'])
            text = seg['text'].strip()
            if not text:
                continue
            lines.append(f"[{timestamp}] {text}")
        return '\n'.join(lines)

    @staticmethod
    def segments_to_srt(segments: List[dict]) -> str:
        lines = []
        for i, seg in enumerate(segments, 1):
            start = SubtitleParser._seconds_to_srt_time(seg['start'])
            end = SubtitleParser._seconds_to_srt_time(seg['end'])
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(seg['text'])
            lines.append("")
        return '\n'.join(lines)

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# =============================================================================
# Groq Whisper 轉錄
# =============================================================================

class GroqTranscriber:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"

    def transcribe(self, audio_path: Path, model: str = "whisper-large-v3") -> Tuple[List[dict], float]:
        file_size = audio_path.stat().st_size
        if file_size > 100 * 1024 * 1024:
            raise ValueError("音檔超過 100MB 限制")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        mime = "audio/mp4" if audio_path.suffix.lower() in [".m4a", ".mp4"] else "audio/mpeg"

        with open(audio_path, "rb") as f:
            files = {
                "file": (audio_path.name, f, mime),
                "model": (None, model),
                "response_format": (None, "verbose_json"),
                "language": (None, "zh"),
            }
            response = requests.post(f"{self.base_url}/audio/transcriptions", headers=headers, files=files, timeout=600)
        if response.status_code != 200:
            raise Exception(f"Groq API 錯誤: {response.text}")

        result = response.json()
        duration = float(result.get("duration", 0) or 0.0)

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                'start': float(seg.get('start', 0) or 0.0),
                'end': float(seg.get('end', 0) or 0.0),
                'text': (seg.get('text', '') or '').strip()
            })
        if not segments and result.get("text"):
            segments.append({'start': 0.0, 'end': duration, 'text': (result.get("text", "") or "").strip()})

        data_manager.add_stt_usage(model, duration)
        return segments, duration

# =============================================================================
# Gemini AI Studio 處理器
# =============================================================================

class GeminiProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def upload_file(self, file_path: Path) -> Any:
        """使用 Files API 上傳檔案"""
        print(f"📤 上傳音檔到 Gemini: {file_path.name}")
        uploaded_file = self.client.files.upload(file=file_path)
        while uploaded_file.state.name == "PROCESSING":
            print("⏳ 等待檔案處理...")
            time.sleep(2)
            uploaded_file = self.client.files.get(name=uploaded_file.name)
        if uploaded_file.state.name == "FAILED":
            raise Exception(f"檔案處理失敗: {uploaded_file.state.name}")
        print(f"✅ 檔案上傳成功: {uploaded_file.name}")
        return uploaded_file

    def process_audio(self, title: str, audio_path: Path, model: str = "google/gemini-3-flash-preview:thinking",
                     segment_minutes: int = 0, enable_query_repeat: bool = False) -> Tuple[str, str, str, int, int]:
        """
        處理音檔，返回：(字幕, 整理內容, 翻譯標題, input_tokens, output_tokens)
        
        Args:
            title: 影片標題
            audio_path: 音訊檔案路徑
            model: 使用的模型
            segment_minutes: 分段時長（分鐘），0 表示不分段
            enable_query_repeat: 是否啟用提詞重複
        """
        from google.genai import types

        # 檢查是否需要分段
        segments_paths = split_audio_by_duration(audio_path, segment_minutes, AUDIO_CACHE_DIR / "segments")
        
        total_input_tokens = 0
        total_output_tokens = 0
        all_subtitles = []
        all_contents = []
        translated_title = title
        
        for idx, segment_path in enumerate(segments_paths):
            print(f"📝 處理音訊段 {idx + 1}/{len(segments_paths)}")
            
            uploaded_file = self.upload_file(segment_path)
            
            # 基礎提示詞
            if idx == 0:
                # 第一段
                prompt = f"""你是一位逐字稿校對員＋內容歸檔整理員。最高優先是「完整保留資訊」，整理內容不是摘要，而是可回放的完整筆記；只移除明顯重複與純口語填充。請嚴格依照指定區塊格式輸出。
        
## 任務一：產生字幕
請將音訊內容轉成逐字稿，格式為：
[MM:SS] 一行內容
[MM:SS] 一行內容
...

要求：
- 每行不要太長，適當斷句
- 保留時間戳記
- 修正明顯錯字

## 任務二：整理內容
將音訊內容整理成結構化的繁體中文文章。

要求：
1. 保留所有重要資訊、知識點、個人看法或心得
2. 如果有漂亮的說法請保留原話
3. 移除重複、口語贅詞、修正錯字
4. 適當分段，加上小標題（使用 ### 標記）
5. 重要概念或關鍵字用 **粗體** 標記
6. 使用繁體中文，專有名詞可原文放在()內
7. 保持原意，不要添加臆測內容
8. 在對應重點那行的開頭保留時間戳 [MM:SS]
9. 盡可能保留資訊量

## 任務三：翻譯標題
將原始標題翻譯成繁體中文（如果已經是中文就保持原樣）

原始標題：{title}

請依照以下格式輸出：

===字幕開始===
（在此輸出字幕）
===字幕結束===

===整理開始===
（在此輸出整理內容）
===整理結束===

===標題===
（在此輸出翻譯後的標題）
===標題結束===
"""
            else:
                # 後續段落，包含前一段的內容作為上下文
                previous_content = all_contents[-1] if all_contents else ""
                prompt = f"""你是一位逐字稿校對員＋內容歸檔整理員。這是音訊的第 {idx + 1} 段，請繼續處理。

##上下文（前一段的整理內容）：
{previous_content[:2000]}
...

## 任務一：產生字幕
請將音訊內容轉成逐字稿，格式為：
[MM:SS] 一行內容

## 任務二：整理內容
繼續整理音訊內容，與前文銜接：
1. 保留所有重要資訊
2. 移除重複、口語贅詞、修正錯字
3. 適當分段，加上小標題（使用 ### 標記）
4. 保留時間戳 [MM:SS]

請依照以下格式輸出：

===字幕開始===
（在此輸出字幕）
===字幕結束===

===整理開始===
（在此輸出整理內容）
===整理結束===
"""
        
            # 轉換模型名稱
            gemini_model = model.replace("google/", "").replace(":thinking", "")
            if not gemini_model.startswith("gemini-"):
                gemini_model = "google/gemini-3-flash-preview"
    
            # 生成配置
            config = types.GenerateContentConfig(
                temperature=1.0,
            )
            
            # 如果是 thinking 模式
            if ":thinking" in model:
                config= types.GenerateContentConfig(
                    temperature=1.0,
                    thinking_config=types.ThinkingConfig(thinking_level="high"),
                )
            
            # 構建 content parts
            parts = [
                types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type),
                types.Part.from_text(text=prompt),
            ]
            
            # 如果啟用提詞重複，將整個 parts 重複一次
            if enable_query_repeat:
                parts = parts + parts
    
            # 呼叫 API
            response = self.client.models.generate_content(
                model=gemini_model,
                contents=[
                    types.Content(
                        role="user",
                        parts=parts
                    )
                ],
                config=config
            )
    
            result_text = ""
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        result_text += part.text
    
            total_input_tokens += response.usage_metadata.prompt_token_count if response.usage_metadata else 0
            total_output_tokens += response.usage_metadata.candidates_token_count if response.usage_metadata else 0
    
            # 刪除上傳的檔案
            try:
                self.client.files.delete(name=uploaded_file.name)
            except:
                pass
    
            # 解析結果
            subtitle_match = re.search(r'===字幕開始===\s*(.*?)\s*===字幕結束===', result_text, re.DOTALL)
            if subtitle_match:
                all_subtitles.append(subtitle_match.group(1).strip())
    
            content_match = re.search(r'===整理開始===\s*(.*?)\s*===整理結束===', result_text, re.DOTALL)
            if content_match:
                all_contents.append(content_match.group(1).strip())
    
            # 只從第一段提取標題
            if idx == 0:
                title_match = re.search(r'===標題===\s*(.*?)\s*===標題結束===', result_text, re.DOTALL)
                if title_match:
                    translated_title = title_match.group(1).strip()
        
        # 清理分段檔案
        if len(segments_paths) > 1:
            for seg_path in segments_paths:
                try:
                    if seg_path.exists() and seg_path != audio_path:
                        seg_path.unlink()
                except:
                    pass
            # 清理分段目錄
            try:
                segments_dir = AUDIO_CACHE_DIR / "segments"
                if segments_dir.exists():
                    import shutil
                    shutil.rmtree(segments_dir, ignore_errors=True)
            except:
                pass
        
        # 合併所有結果
        final_subtitle = "\n\n".join(all_subtitles)
        final_content = "\n\n".join(all_contents)
        
        return final_subtitle, final_content, translated_title, total_input_tokens, total_output_tokens

    def summarize_text(self, title: str, content: str, model: str = "google/gemini-3-flash-preview:thinking") -> Tuple[str, str, int, int]:
        """整理文字內容，返回：(翻譯標題, 整理內容, input_tokens, output_tokens)"""
        from google.genai import types

        prompt = f"""你是一位逐字稿校對員＋內容歸檔整理員。最高優先是「完整保留資訊」，整理內容不是摘要，而是可回放的完整筆記；只移除明顯重複與純口語填充。請嚴格依照指定區塊格式輸出。

要求：
1. 保留所有重要資訊、知識點、個人看法或心得
2. 如果有漂亮的說法請保留原話
3. 移除重複、口語贅詞、修正錯字
4. 適當分段，加上小標題（使用 ### 標記）
5. 重要概念或關鍵字用 **粗體** 標記
6. 使用繁體中文，專有名詞可原文放在()內
7. 保持原意，不要添加臆測內容
8. 如果有時間戳標記 [MM:SS] 或 [HH:MM:SS]，請保留在對應重點那行的開頭
9. 盡可能保留資訊量
10. 最後一行請將原始標題翻譯成繁體中文，格式為：【標題】你的標題

原始標題：{title}

字幕內容：
{content}

請開始整理："""

        gemini_model = model.replace("google/", "").replace(":thinking", "")
        if not gemini_model.startswith("gemini-"):
            gemini_model = "google/gemini-3-flash-preview"

        config = types.GenerateContentConfig(
            temperature=1.0,
        )
        
        if ":thinking" in model:
            config = types.GenerateContentConfig(
                temperature=1.0,
                thinking_config=types.ThinkingConfig(thinking_level="high"),
            )

        response = self.client.models.generate_content(
            model=gemini_model,
            contents=prompt,
            config=config
        )

        result = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    result += part.text

        input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0

        # 提取標題
        generated_title = ""
        lines = result.strip().split('\n')
        for i, line in enumerate(lines):
            if line.startswith('【標題】'):
                generated_title = line.replace('【標題】', '').strip()
                lines.pop(i)
                break

        content_result = '\n'.join(lines).strip()
        if not generated_title:
            generated_title = title[:120]

        return generated_title, content_result, input_tokens, output_tokens

# =============================================================================
# 任務處理
# =============================================================================

# 正在整理中的 job
summarizing_jobs: Set[str] = set()
# 正在處理的整理任務
active_summarize_tasks: Dict[str, threading.Event] = {}
batch_tasks_lock = threading.Lock()
batch_tasks: Dict[str, Dict[str, Any]] = {}
APP_PORT = 5000

class JobProcessor:
    def __init__(self):
        self.ytdlp = YTDLPProcessor()

    def _refresh_and_check_cancel(self, job: Job):
        latest = data_manager.get_job(job.id)
        if not latest:
            raise Exception("任務不存在（可能已刪除）")
        if getattr(latest, "deleted", False):
            raise Exception("任務已刪除")
        if getattr(latest, "cancel_requested", False) or latest.status == "cancelled":
            raise Exception("使用者已取消")

    def _set_stage(self, job: Job, stage: str, progress: int = None, status: str = None):
        latest = data_manager.get_job(job.id)
        if not latest or getattr(latest, "deleted", False):
            return
        latest.stage = stage
        if progress is not None:
            latest.progress = int(progress)
        if status is not None:
            latest.status = status
        data_manager.update_job(latest)
        job.stage = latest.stage
        job.progress = latest.progress
        job.status = latest.status

    def process_job(self, job_id: str):
        job = data_manager.get_job(job_id)
        if not job or job.deleted:
            return

        config = data_manager.config
        audio_format = config.get("audio_format", "m4a")
        no_subtitle_action = config.get("no_subtitle_action", "llm_direct")
        download_video = config.get("download_video", False)

        try:
            self._set_stage(job, "解析網址/初始化", progress=3, status="downloading")
            self._refresh_and_check_cancel(job)

            platform, video_id = extract_video_info(job.url)
            job.platform = platform
            job.video_id = video_id
            data_manager.update_job(job)

            self._set_stage(job, "取得影片資訊", progress=10)
            info = self.ytdlp.get_video_info(job.url)
            self._refresh_and_check_cancel(job)

            job.title = info.get("title", "") or info.get("fulltitle", "") or "未知標題"
            job.channel = info.get("channel", "") or info.get("uploader", "") or ""
            job.uploader = info.get("uploader", "") or info.get("channel", "") or ""
            job.upload_date = info.get("upload_date", "") or ""
            job.duration = info.get("duration", 0) or 0
            data_manager.update_job(job)

            # 下載字幕
            self._set_stage(job, "下載字幕", progress=25)
            subtitle_result = self.ytdlp.download_subtitle(job.url, SUBTITLE_DIR, job.title, video_id, info=info)
            self._refresh_and_check_cancel(job)

            if subtitle_result:
                subtitle_path, _lang = subtitle_result
                job.subtitle_path = str(subtitle_path)
                job.has_original_subtitle = True
                content = subtitle_path.read_text(encoding="utf-8", errors="ignore")
                if subtitle_path.suffix == ".vtt":
                    segments = SubtitleParser.parse_vtt(content)
                else:
                    segments = SubtitleParser.parse_srt(content)
                job.subtitle_content = SubtitleParser.segments_to_text(segments)
                job.subtitle_with_time = SubtitleParser.segments_to_timestamped_text(segments, platform, video_id, job.url)

                # 下載影片（如果開啟）
                if download_video:
                    self._set_stage(job, "下載原影片", progress=80)
                    video_path = self.ytdlp.download_video(job.url, VIDEO_DIR, job.title, video_id)
                    if video_path:
                        job.video_path = str(video_path)

                job.progress = 100
                job.stage = "完成（字幕）"
                job.status = "completed"
                data_manager.update_job(job)
                return

            # 無字幕 → 根據設定處理
            if no_subtitle_action == "audio_only":
                # 只下載音軌
                self._set_stage(job, "無字幕，下載音軌", progress=50, status="downloading")
                audio_path = self.ytdlp.download_audio(job.url, AUDIO_CACHE_DIR, job.title, video_id, audio_format)
                if audio_path:
                    job.audio_path = str(audio_path)
                if download_video:
                    self._set_stage(job, "下載原影片", progress=80)
                    video_path = self.ytdlp.download_video(job.url, VIDEO_DIR, job.title, video_id)
                    if video_path:
                        job.video_path = str(video_path)
                job.progress = 100
                job.stage = "完成（僅音軌）"
                job.status = "completed"
                data_manager.update_job(job)
                return

            # 下載音軌
            self._set_stage(job, "無字幕，下載音軌", progress=40, status="downloading")
            audio_path = self.ytdlp.download_audio(job.url, AUDIO_CACHE_DIR, job.title, video_id, audio_format)
            self._refresh_and_check_cancel(job)

            if not audio_path:
                detail = (self.ytdlp.last_audio_error or "").strip()
                if detail:
                    raise Exception(f"無法下載音軌: {detail}")
                raise Exception("無法下載音軌")

            job.audio_path = str(audio_path)
            data_manager.update_job(job)

            if no_subtitle_action == "llm_direct":
                # 直接送 LLM（完成時不做轉錄，整理時才處理）
                if download_video:
                    self._set_stage(job, "下載原影片", progress=80)
                    video_path = self.ytdlp.download_video(job.url, VIDEO_DIR, job.title, video_id)
                    if video_path:
                        job.video_path = str(video_path)
                job.progress = 100
                job.stage = "完成（音軌，待 AI 整理）"
                job.status = "completed"
                data_manager.update_job(job)
                return

            # STT 流程
            groq_key = config.get("groq_api_key", "")
            if not groq_key:
                raise Exception("未設定 Groq API Key，無法進行 STT")

            self._set_stage(job, "音訊前處理", progress=55, status="transcribing")
            cache_prefix = get_url_hash(job.url)
            speech_enhance = config.get("speech_enhance_preset", "strong")
            noise_db = float(config.get("silence_noise_db", -40) or -40)
            min_dur = float(config.get("silence_min_duration", 1.0) or 1.0)
            speed = float(config.get("llm_audio_speed", 1.5) or 1.5)

            processed_audio, time_map = AudioPreprocessor.preprocess(
                Path(job.audio_path), cache_prefix,
                speech_enhance_preset=speech_enhance,
                noise_db=noise_db, min_duration=min_dur, speed=speed
            )
            self._refresh_and_check_cancel(job)

            self._set_stage(job, "語音轉文字（STT）", progress=70, status="transcribing")
            stt_model = config.get("default_stt_model", "whisper-large-v3")
            transcriber = GroqTranscriber(groq_key)
            segments_processed, duration = transcriber.transcribe(processed_audio, stt_model)

            # 補償時間戳
            segments = []
            for seg in segments_processed:
                s_orig = remap_timestamp_seconds(seg["start"], speed, time_map)
                e_orig = remap_timestamp_seconds(seg["end"], speed, time_map)
                segments.append({"start": s_orig, "end": e_orig, "text": seg.get("text", "").strip()})

            job.subtitle_content = SubtitleParser.segments_to_text(segments)
            job.subtitle_with_time = SubtitleParser.segments_to_timestamped_text(segments, platform, video_id, job.url)

            # 儲存字幕檔
            safe_title = sanitize_filename(job.title)
            srt_path = SUBTITLE_DIR / f"{safe_title}_{video_id}_transcribed.srt"
            srt_path.write_text(SubtitleParser.segments_to_srt(segments), encoding="utf-8")
            job.subtitle_path = str(srt_path)

            # 清理暫存
            AudioPreprocessor.cleanup_cache(cache_prefix)

            if download_video:
                self._set_stage(job, "下載原影片", progress=90)
                video_path = self.ytdlp.download_video(job.url, VIDEO_DIR, job.title, video_id)
                if video_path:
                    job.video_path = str(video_path)

            job.progress = 100
            job.stage = "完成（轉錄）"
            job.status = "completed"
            data_manager.update_job(job)

        except Exception as e:
            latest = data_manager.get_job(job_id)
            if latest and latest.deleted:
                return
            if latest and (latest.cancel_requested or latest.status == "cancelled"):
                latest.status = "cancelled"
                latest.stage = "已取消"
                latest.error_message = "使用者已取消"
                latest.progress = min(latest.progress, 99)
                data_manager.update_job(latest)
                return

            job = data_manager.get_job(job_id) or job
            if job and not job.deleted:
                job.status = "error"
                job.stage = "錯誤"
                job.error_message = str(e)
                data_manager.update_job(job)

job_processor = JobProcessor()

job_queue: "queue.Queue[str]" = queue.Queue()
worker_threads: List[threading.Thread] = []
worker_stop = threading.Event()
MAX_EXTRACTION_WORKERS = 3

def worker_loop():
    while not worker_stop.is_set():
        try:
            job_id = job_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            job_processor.process_job(job_id)
        finally:
            job_queue.task_done()

def start_worker_pool(worker_count: int = MAX_EXTRACTION_WORKERS):
    target = max(1, min(MAX_EXTRACTION_WORKERS, int(worker_count or 1)))
    alive = [t for t in worker_threads if t.is_alive()]
    worker_threads[:] = alive
    while len(worker_threads) < target:
        t = threading.Thread(target=worker_loop, daemon=True)
        t.start()
        worker_threads.append(t)

def expand_urls_for_jobs(urls: List[str], ytdlp: Optional[YTDLPProcessor] = None) -> List[str]:
    processor = ytdlp or YTDLPProcessor()
    expanded: List[str] = []
    seen: Set[str] = set()

    def add_unique(u: str):
        key = (u or "").strip()
        if not key:
            return
        if key in seen:
            return
        seen.add(key)
        expanded.append(key)

    for raw in urls:
        url = (raw or "").strip()
        if not url:
            continue
        lowered = url.lower()
        looks_like_playlist = (
            "list=" in lowered or
            "/playlist" in lowered or
            "playlist?" in lowered or
            "/sets/" in lowered or
            "favlist" in lowered
        )
        if looks_like_playlist:
            playlist_urls = processor.get_playlist_urls(url)
            if playlist_urls:
                for pu in playlist_urls:
                    add_unique(pu)
            else:
                add_unique(url)
        else:
            add_unique(url)
    return expanded

def enqueue_job(url: str) -> Job:
    job = Job(id=str(uuid.uuid4()), url=url, status="queued", stage="等待處理", progress=0)
    data_manager.add_job(job)
    job_queue.put(job.id)
    start_worker_pool()
    return job

def create_job(url: str) -> Job:
    job = Job(id=str(uuid.uuid4()), url=url, status="queued", stage="等待處理", progress=0)
    data_manager.add_job(job)
    return job

def _batch_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "total": len(items),
        "pending": sum(1 for i in items if i.get("status") == "pending"),
        "extracting": sum(1 for i in items if i.get("status") == "extracting"),
        "summarizing": sum(1 for i in items if i.get("status") == "summarizing"),
        "completed": sum(1 for i in items if i.get("status") == "completed"),
        "error": sum(1 for i in items if i.get("status") == "error"),
    }

def _batch_update_item(batch_id: str, item_id: str, **fields):
    with batch_tasks_lock:
        task = batch_tasks.get(batch_id)
        if not task:
            return
        for item in task["items"]:
            if item["id"] == item_id:
                item.update(fields)
                break
        counts = _batch_counts(task["items"])
        task["counts"] = counts
        if task["status"] == "running" and counts["completed"] + counts["error"] >= counts["total"]:
            task["status"] = "completed"
            task["finished_at"] = datetime.now().isoformat()

def _batch_snapshot(batch_id: str) -> Optional[Dict[str, Any]]:
    with batch_tasks_lock:
        task = batch_tasks.get(batch_id)
        if not task:
            return None
        return {
            "id": task["id"],
            "status": task["status"],
            "created_at": task["created_at"],
            "finished_at": task.get("finished_at", ""),
            "concurrency": task["concurrency"],
            "counts": dict(task.get("counts", {})),
            "items": [dict(item) for item in task["items"]],
        }

def _extract_batch_item(batch_id: str, item: Dict[str, Any]) -> Optional[Dict[str, str]]:
    item_id = item["id"]
    category = (item.get("category") or "未分類").strip() or "未分類"
    try:
        _batch_update_item(batch_id, item_id, status="extracting", message="")
        job = create_job(item["url"])
        _batch_update_item(batch_id, item_id, job_id=job.id)
        job_processor.process_job(job.id)
        latest = data_manager.get_job(job.id)
        if not latest:
            _batch_update_item(batch_id, item_id, status="error", message="找不到工作")
            return None

        if latest.status != "completed":
            _batch_update_item(batch_id, item_id, status="error", message=latest.error_message or latest.status or "提取失敗")
            return None

        _batch_update_item(batch_id, item_id, status="summarizing", message="")
        return {
            "item_id": item_id,
            "job_id": job.id,
            "category": category,
        }
    except Exception as e:
        _batch_update_item(batch_id, item_id, status="error", message=str(e))
        return None

def _summarize_batch_item(batch_id: str, item_meta: Dict[str, str]):
    item_id = item_meta["item_id"]
    job_id = item_meta["job_id"]
    category = item_meta["category"]
    try:
        task_id = f"batch_{batch_id}_{item_id}_{int(time.time() * 1000)}"
        res = requests.post(
            f"http://127.0.0.1:{APP_PORT}/api/summaries",
            json={"job_id": job_id, "task_id": task_id},
            timeout=7200
        )
        try:
            payload = res.json()
        except Exception:
            payload = {"error": f"摘要回傳格式錯誤 ({res.status_code})"}

        if res.status_code >= 400 or payload.get("error"):
            _batch_update_item(batch_id, item_id, status="error", message=payload.get("error", f"摘要失敗 ({res.status_code})"))
            return

        summary_id = payload.get("id", "")
        if summary_id:
            data_manager.move_summary(summary_id, category)

        _batch_update_item(batch_id, item_id, status="completed", summary_id=summary_id, message="")
    except Exception as e:
        _batch_update_item(batch_id, item_id, status="error", message=str(e))

def _run_batch_task(batch_id: str):
    with batch_tasks_lock:
        task = batch_tasks.get(batch_id)
        if not task:
            return
        items = [dict(item) for item in task["items"]]
        concurrency = int(task.get("concurrency", 3) or 3)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for item in items:
            item_meta = _extract_batch_item(batch_id, item)
            if item_meta:
                futures.append(executor.submit(_summarize_batch_item, batch_id, item_meta))
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception:
                pass

    with batch_tasks_lock:
        task = batch_tasks.get(batch_id)
        if task and task["status"] != "completed":
            task["counts"] = _batch_counts(task["items"])
            task["status"] = "completed"
            task["finished_at"] = datetime.now().isoformat()

# =============================================================================
# Flask
# =============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'subtitle-tool-v30'

@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/batch')
def batch_page():
    return render_template_string(BATCH_TEMPLATE)

@app.route('/api/batch/start', methods=['POST'])
def start_batch():
    data = request.json or {}
    raw_items = data.get("items", [])
    try:
        concurrency = int(data.get("concurrency", 3))
    except Exception:
        concurrency = 3
    concurrency = max(1, min(3, concurrency))

    processor = YTDLPProcessor()
    parsed_items: List[Dict[str, Any]] = []
    for raw_item in raw_items:
        category = (raw_item.get("category") or "未分類").strip() or "未分類"
        urls = raw_item.get("urls", [])
        if isinstance(urls, str):
            urls = urls.splitlines()
        expanded = expand_urls_for_jobs(urls, processor)
        for url in expanded:
            parsed_items.append({
                "id": str(uuid.uuid4()),
                "category": category,
                "url": url,
                "status": "pending",
                "message": "",
                "job_id": "",
                "summary_id": "",
            })

    if not parsed_items:
        return jsonify({"error": "請至少填入一個有效網址"}), 400

    batch_id = str(uuid.uuid4())
    task = {
        "id": batch_id,
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "finished_at": "",
        "concurrency": concurrency,
        "items": parsed_items,
    }
    task["counts"] = _batch_counts(task["items"])

    with batch_tasks_lock:
        batch_tasks[batch_id] = task

    threading.Thread(target=_run_batch_task, args=(batch_id,), daemon=True).start()
    return jsonify({"batch_id": batch_id, "counts": task["counts"], "concurrency": concurrency})

@app.route('/api/batch/<batch_id>', methods=['GET'])
def get_batch(batch_id):
    snapshot = _batch_snapshot(batch_id)
    if not snapshot:
        return jsonify({"error": "找不到批次任務"}), 404
    return jsonify(snapshot)

@app.route('/api/config', methods=['GET'])
def get_config():
    cfg = data_manager.config.copy()
    # 隱藏 API Key 細節
    has_gemini_key = bool(cfg.get("gemini_api_key"))
    has_groq_key = bool(cfg.get("groq_api_key"))
    cfg.pop("gemini_api_key", None)
    cfg.pop("groq_api_key", None)
    cfg["has_gemini_key"] = has_gemini_key
    cfg["has_groq_key"] = has_groq_key
    return jsonify(cfg)

@app.route('/api/config', methods=['PUT'])
def update_config():
    data = request.json or {}

    # API Keys
    if "gemini_api_key" in data:
        data_manager.config["gemini_api_key"] = data["gemini_api_key"].strip()
    if "groq_api_key" in data:
        data_manager.config["groq_api_key"] = data["groq_api_key"].strip()

    # 模型設定
    if "default_llm_model" in data:
        data_manager.config["default_llm_model"] = data["default_llm_model"]
    if "default_stt_model" in data:
        data_manager.config["default_stt_model"] = data["default_stt_model"]

    # 無字幕處理
    if "no_subtitle_action" in data:
        data_manager.config["no_subtitle_action"] = data["no_subtitle_action"]

    # 音檔格式
    if "audio_format" in data:
        data_manager.config["audio_format"] = data["audio_format"]

    # 人聲加強
    if "speech_enhance_preset" in data:
        data_manager.config["speech_enhance_preset"] = data["speech_enhance_preset"]

    # 其他設定
    def to_float(v, default):
        try:
            return float(v)
        except:
            return float(default)

    def to_int(v, default):
        try:
            return int(v)
        except:
            return int(default)

    def to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    if "llm_audio_speed" in data:
        data_manager.config["llm_audio_speed"] = to_float(data["llm_audio_speed"], 1.5)
    if "silence_noise_db" in data:
        data_manager.config["silence_noise_db"] = to_float(data["silence_noise_db"], -40)
    if "silence_min_duration" in data:
        data_manager.config["silence_min_duration"] = to_float(data["silence_min_duration"], 1.0)
    if "long_video_threshold_minutes" in data:
        data_manager.config["long_video_threshold_minutes"] = to_float(data["long_video_threshold_minutes"], 30)
    if "download_video" in data:
        data_manager.config["download_video"] = to_bool(data["download_video"])
    if "audio_segment_minutes" in data:
        data_manager.config["audio_segment_minutes"] = max(0, to_int(data["audio_segment_minutes"], 0))
    if "enable_query_repeat" in data:
        data_manager.config["enable_query_repeat"] = to_bool(data["enable_query_repeat"])

    data_manager._save_data()
    return jsonify({"success": True})

@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    jobs = []
    for j in data_manager.jobs.values():
        if getattr(j, "deleted", False):
            continue
        job_dict = asdict(j)
        job_dict["is_summarizing"] = j.id in summarizing_jobs
        jobs.append(job_dict)
    return jsonify(jobs)

@app.route('/api/jobs/check-duration', methods=['POST'])
def check_job_duration():
    data = request.json or {}
    urls = data.get('urls', [])
    threshold = data_manager.config.get("long_video_threshold_minutes", 30) * 60

    ytdlp = YTDLPProcessor()
    results = []

    for url in urls:
        url = (url or "").strip()
        if not url:
            continue
        try:
            info = ytdlp.get_video_info(url)
            duration = info.get("duration", 0) or 0
            title = info.get("title", "") or "未知標題"
            results.append({
                "url": url,
                "title": title,
                "duration": duration,
                "duration_str": format_timestamp(duration),
                "needs_confirmation": duration > threshold
            })
        except:
            results.append({
                "url": url,
                "title": "無法取得資訊",
                "duration": 0,
                "duration_str": "00:00",
                "needs_confirmation": False
            })

    return jsonify(results)

@app.route('/api/jobs', methods=['POST'])
def create_jobs():
    data = request.json or {}
    urls = data.get('urls', [])
    all_expanded_urls = expand_urls_for_jobs(urls)

    created_jobs = []
    for url in all_expanded_urls:
        job = enqueue_job(url)
        created_jobs.append(asdict(job))
    return jsonify(created_jobs)

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    job = data_manager.get_job(job_id)
    if not job or job.deleted:
        return jsonify({'error': '任務不存在'}), 404
    job_dict = asdict(job)
    job_dict["is_summarizing"] = job_id in summarizing_jobs
    return jsonify(job_dict)

@app.route('/api/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    data_manager.soft_delete_job(job_id)
    return jsonify({'success': True})

@app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    job = data_manager.get_job(job_id)
    if not job or job.deleted:
        return jsonify({'error': '任務不存在'}), 404
    job.cancel_requested = True
    job.status = "cancelled"
    job.stage = "已取消"
    job.error_message = "使用者已取消"
    data_manager.update_job(job)
    return jsonify({'success': True})

@app.route('/api/jobs/<job_id>/llm-input', methods=['GET'])
def get_job_llm_input(job_id):
    """取得複製指令的內容"""
    job = data_manager.get_job(job_id)
    if not job or job.deleted:
        return jsonify({'error': '任務不存在'}), 404

    video_title = job.title

    if job.subtitle_content:
        # 有字幕：複製整理指令 + 字幕
        content_for_summary = job.subtitle_with_time or job.subtitle_content
        prompt = f"""你是一位會議紀要與內容歸檔整理員。最高優先是「完整保留資訊」，不要為了精簡而刪掉例子、數字、條件、因果、反例、前提、定義、步驟、對比、結論與但書；只移除明顯重複與純口語填充。請嚴格遵守輸出格式。 請將以下字幕內容整理成結構化的繁體中文文章。

要求：
1. 保留所有重要資訊、知識點、個人看法或心得
2. 如果有漂亮的說法請保留原話
3. 移除重複、口語贅詞、修正錯字
4. 適當分段，加上小標題（使用 ### 標記）
5. 重要概念或關鍵字用 **粗體** 標記
6. 使用繁體中文，專有名詞可原文放在()內
7. 保持原意，不要添加臆測內容
8. 如果有時間戳標記 [MM:SS] 或 [HH:MM:SS]，請保留在對應重點那行的開頭
9. 盡可能保留資訊量
10. 最後一行請將原始標題翻譯成繁體中文，格式為：【標題】你的標題

原始標題：{video_title}

字幕內容：
{content_for_summary}

請開始整理："""
        return jsonify({'content': prompt})
    elif job.audio_path:
        # 無字幕只有音檔：複製音檔處理指令
        prompt = f"""你是一位逐字稿校對員＋內容歸檔整理員。最高優先是「完整保留資訊」，整理內容不是摘要，而是可回放的完整筆記；只移除明顯重複與純口語填充。請嚴格依照指定區塊格式輸出。
        
## 任務一：產生字幕
請將音訊內容轉成逐字稿，格式為：
[MM:SS] 一行內容
[MM:SS] 一行內容
...

要求：
- 每行不要太長，適當斷句
- 保留時間戳記
- 修正明顯錯字

## 任務二：整理內容
將音訊內容整理成結構化的繁體中文文章。

要求：
1. 保留所有重要資訊、知識點、個人看法或心得
2. 如果有漂亮的說法請保留原話
3. 移除重複、口語贅詞、修正錯字
4. 適當分段，加上小標題（使用 ### 標記）
5. 重要概念或關鍵字用 **粗體** 標記
6. 使用繁體中文，專有名詞可原文放在()內
7. 保持原意，不要添加臆測內容
8. 在對應重點那行的開頭保留時間戳 [MM:SS]
9. 盡可能保留資訊量

## 任務三：翻譯標題
將原始標題翻譯成繁體中文（如果已經是中文就保持原樣）

原始標題：{video_title}

請依照以下格式輸出：

===字幕開始===
（在此輸出字幕）
===字幕結束===

===整理開始===
（在此輸出整理內容）
===整理結束===

===標題===
（在此輸出翻譯後的標題）
===標題結束===
"""
        return jsonify({'content': prompt, 'is_audio': True})
    else:
        return jsonify({'error': '無字幕內容且無音檔'}), 400

@app.route('/api/jobs/<job_id>/audio', methods=['GET'])
def get_job_audio(job_id):
    job = data_manager.get_job(job_id)
    if not job or job.deleted:
        return jsonify({'error': '任務不存在'}), 404
    if not job.audio_path:
        return jsonify({'error': '無音檔'}), 404

    audio_path = Path(job.audio_path)
    if not audio_path.exists():
        return jsonify({'error': '音檔不存在'}), 404

    return send_file(
        audio_path,
        mimetype='audio/mp4',
        as_attachment=True,
        download_name=audio_path.name
    )

@app.route('/api/jobs/<job_id>/video', methods=['GET'])
def get_job_video(job_id):
    job = data_manager.get_job(job_id)
    if not job or job.deleted:
        return jsonify({'error': '任務不存在'}), 404
    if not job.video_path:
        return jsonify({'error': '無影片'}), 404

    video_path = Path(job.video_path)
    if not video_path.exists():
        return jsonify({'error': '影片不存在'}), 404

    return send_file(
        video_path,
        mimetype='video/mp4',
        as_attachment=True,
        download_name=video_path.name
    )

@app.route('/api/jobs/<job_id>/subtitle', methods=['GET'])
def get_job_subtitle(job_id):
    job = data_manager.get_job(job_id)
    if not job or job.deleted:
        return jsonify({'error': '任務不存在'}), 404
    if not job.subtitle_path:
        return jsonify({'error': '無字幕'}), 404

    subtitle_path = Path(job.subtitle_path)
    if not subtitle_path.exists():
        return jsonify({'error': '字幕檔不存在'}), 404

    return send_file(
        subtitle_path,
        mimetype='text/plain',
        as_attachment=True,
        download_name=subtitle_path.name
    )

@app.route('/api/summaries', methods=['GET'])
def get_summaries():
    return jsonify([asdict(s) for s in data_manager.summaries.values()])

@app.route('/api/summaries', methods=['POST'])
def create_summary():
    data = request.json or {}
    job_id = data.get('job_id')
    task_id = data.get('task_id', str(uuid.uuid4()))

    job = data_manager.get_job(job_id)
    if not job or job.deleted:
        return jsonify({'error': '無效的任務'}), 400

    config = data_manager.config
    gemini_key = config.get("gemini_api_key", "")
    if not gemini_key:
        return jsonify({'error': '未設定 Gemini API Key'}), 400

    has_text = bool(job.subtitle_content)
    has_audio = bool(job.audio_path)

    if not has_text and not has_audio:
        return jsonify({'error': '此任務沒有字幕也沒有音檔'}), 400

    # 標記為整理中
    summarizing_jobs.add(job_id)
    cancel_event = threading.Event()
    active_summarize_tasks[task_id] = cancel_event

    try:
        gemini = GeminiProcessor(gemini_key)
        model = config.get("default_llm_model", "google/gemini-3-flash-preview:thinking")
        final_title = job.title

        if has_text:
            # 有字幕：整理字幕
            content_for_summary = job.subtitle_with_time or job.subtitle_content

            if cancel_event.is_set():
                return jsonify({'error': '已取消', 'cancelled': True}), 200

            generated_title, content, input_tokens, output_tokens = gemini.summarize_text(
                final_title, content_for_summary, model
            )
            content_with_time = linkify_timestamps_in_text(content, job.platform, job.video_id, job.url)

            data_manager.add_usage_record(
                model, input_tokens, output_tokens,
                description=f"整理字幕: {final_title[:30]}..."
            )
        else:
            # 無字幕：處理音檔
            if cancel_event.is_set():
                return jsonify({'error': '已取消', 'cancelled': True}), 200

            # 前處理音檔
            cache_prefix = get_url_hash(job.url)
            speech_enhance = config.get("speech_enhance_preset", "strong")
            noise_db = float(config.get("silence_noise_db", -40) or -40)
            min_dur = float(config.get("silence_min_duration", 1.0) or 1.0)
            speed = float(config.get("llm_audio_speed", 1.5) or 1.5)

            processed_audio, time_map = AudioPreprocessor.preprocess(
                Path(job.audio_path), cache_prefix,
                speech_enhance_preset=speech_enhance,
                noise_db=noise_db, min_duration=min_dur, speed=speed
            )

            if cancel_event.is_set():
                AudioPreprocessor.cleanup_cache(cache_prefix)
                return jsonify({'error': '已取消', 'cancelled': True}), 200

            # 獲取配置參數
            segment_minutes = int(config.get("audio_segment_minutes", 0) or 0)
            enable_query_repeat = bool(config.get("enable_query_repeat", False))
            
            subtitle, content, generated_title, input_tokens, output_tokens = gemini.process_audio(
                final_title, processed_audio, model,
                segment_minutes=segment_minutes,
                enable_query_repeat=enable_query_repeat
            )

            # 補償時間戳
            subtitle = remap_timestamps_in_text(subtitle, speed, time_map)
            content = remap_timestamps_in_text(content, speed, time_map)

            content_with_time = linkify_timestamps_in_text(content, job.platform, job.video_id, job.url)

            # 更新 job 的字幕欄位
            job.subtitle_content = subtitle
            job.subtitle_with_time = subtitle

            # 儲存字幕檔
            safe_title = sanitize_filename(job.title)
            srt_lines = []
            for i, line in enumerate(subtitle.split('\n'), 1):
                match = re.match(r'\[(\d+:\d+(?::\d+)?)\]\s*(.+)', line)
                if match:
                    ts, text = match.groups()
                    sec = time_str_to_seconds(ts)
                    start = SubtitleParser._seconds_to_srt_time(sec)
                    end = SubtitleParser._seconds_to_srt_time(sec + 3)
                    srt_lines.extend([str(i), f"{start} --> {end}", text, ""])
            if srt_lines:
                srt_path = SUBTITLE_DIR / f"{safe_title}_{job.video_id}_llm.srt"
                srt_path.write_text('\n'.join(srt_lines), encoding="utf-8")
                job.subtitle_path = str(srt_path)

            data_manager.update_job(job)

            # 記錄使用量
            audio_duration = get_audio_duration_seconds(processed_audio)

            # 清理暫存
            AudioPreprocessor.cleanup_cache(cache_prefix)

            data_manager.add_usage_record(
                model, input_tokens, output_tokens, audio_seconds=audio_duration,
                description=f"整理音訊: {final_title[:30]}..."
            )

        if not generated_title:
            generated_title = final_title

        summary = Summary(
            id=str(uuid.uuid4()),
            job_id=job_id,
            title=generated_title,
            content=content,
            content_with_time=content_with_time,
            video_url=job.url,
            video_title=job.title,
            channel=job.channel,
            uploader=job.uploader,
            upload_date=job.upload_date,
            model_used=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        data_manager.add_summary(summary)
        return jsonify(asdict(summary))

    finally:
        summarizing_jobs.discard(job_id)
        if task_id in active_summarize_tasks:
            del active_summarize_tasks[task_id]

@app.route('/api/summaries/cancel/<task_id>', methods=['POST'])
def cancel_summarize(task_id):
    if task_id in active_summarize_tasks:
        active_summarize_tasks[task_id].set()
        return jsonify({'success': True})
    return jsonify({'error': '任務不存在或已完成'}), 404

@app.route('/api/summaries/manual', methods=['POST'])
def create_summary_manual():
    """手動匯入整理結果"""
    data = request.json or {}
    job_id = data.get("job_id")
    raw = (data.get("raw_output") or "").strip()
    model = (data.get("model") or "manual").strip()

    job = data_manager.get_job(job_id)
    if not job or job.deleted:
        return jsonify({'error': '無效任務'}), 400
    if not raw:
        return jsonify({'error': '請貼上整理結果'}), 400

    # 解析輸入（支援格式化輸出或純文字）
    subtitle = ""
    content = ""
    generated_title = ""

    # 嘗試解析格式化輸出
    subtitle_match = re.search(r'===字幕開始===\s*(.*?)\s*===字幕結束===', raw, re.DOTALL)
    if subtitle_match:
        subtitle = subtitle_match.group(1).strip()

    content_match = re.search(r'===整理開始===\s*(.*?)\s*===整理結束===', raw, re.DOTALL)
    if content_match:
        content = content_match.group(1).strip()

    title_match = re.search(r'===標題===\s*(.*?)\s*===標題結束===', raw, re.DOTALL)
    if title_match:
        generated_title = title_match.group(1).strip()

    # 如果沒有格式化，嘗試解析【標題】格式
    if not content:
        lines = raw.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('【標題】'):
                generated_title = line.replace('【標題】', '').strip()
                lines.pop(i)
                break
        content = '\n'.join(lines).strip()

    if not generated_title:
        generated_title = job.title[:120]

    # 更新字幕到 job
    if subtitle:
        job.subtitle_content = subtitle
        job.subtitle_with_time = subtitle
        # 儲存字幕檔
        safe_title = sanitize_filename(job.title)
        srt_lines = []
        for i, line in enumerate(subtitle.split('\n'), 1):
            match = re.match(r'\[(\d+:\d+(?::\d+)?)\]\s*(.+)', line)
            if match:
                ts, text = match.groups()
                sec = time_str_to_seconds(ts)
                start = SubtitleParser._seconds_to_srt_time(sec)
                end = SubtitleParser._seconds_to_srt_time(sec + 3)
                srt_lines.extend([str(i), f"{start} --> {end}", text, ""])
        if srt_lines:
            srt_path = SUBTITLE_DIR / f"{safe_title}_{job.video_id}_manual.srt"
            srt_path.write_text('\n'.join(srt_lines), encoding="utf-8")
            job.subtitle_path = str(srt_path)
        data_manager.update_job(job)

    content_with_time = linkify_timestamps_in_text(content, job.platform, job.video_id, job.url)

    summary = Summary(
        id=str(uuid.uuid4()),
        job_id=job_id,
        title=generated_title[:120],
        content=content,
        content_with_time=content_with_time,
        video_url=job.url,
        video_title=job.title,
        channel=job.channel,
        uploader=job.uploader,
        upload_date=job.upload_date,
        model_used=model,
        input_tokens=0,
        output_tokens=0
    )
    data_manager.add_summary(summary)
    return jsonify(asdict(summary))

@app.route('/api/summaries/<summary_id>', methods=['DELETE'])
def delete_summary(summary_id):
    if data_manager.delete_summary(summary_id):
        return jsonify({'success': True})
    return jsonify({'error': '刪除失敗'}), 400

@app.route('/api/summaries/batch-delete', methods=['POST'])
def batch_delete_summaries():
    data = request.json or {}
    ids = data.get('ids', [])
    data_manager.delete_summaries_batch(ids)
    return jsonify({'success': True})

@app.route('/api/summaries/batch-move', methods=['POST'])
def batch_move_summaries():
    data = request.json or {}
    ids = data.get('ids', [])
    category = data.get('category', '未分類')
    data_manager.move_summaries_batch(ids, category)
    return jsonify({'success': True})

@app.route('/api/summaries/<summary_id>/pin', methods=['POST'])
def toggle_pin(summary_id):
    if data_manager.toggle_pin_summary(summary_id):
        summary = data_manager.summaries.get(summary_id)
        return jsonify({'success': True, 'pinned': summary.pinned if summary else False})
    return jsonify({'error': '操作失敗'}), 400

@app.route('/api/summaries/<summary_id>/title', methods=['PUT'])
def update_title(summary_id):
    data = request.json or {}
    new_title = (data.get('title') or '').strip()
    if new_title and data_manager.update_summary_title(summary_id, new_title):
        return jsonify({'success': True})
    return jsonify({'error': '更新失敗'}), 400

@app.route('/api/summaries/<summary_id>/content', methods=['PUT'])
def update_content(summary_id):
    data = request.json or {}
    new_content = (data.get('content') or '').strip()
    if new_content and data_manager.update_summary_content(summary_id, new_content):
        return jsonify({'success': True})
    return jsonify({'error': '更新失敗'}), 400

@app.route('/api/summaries/<summary_id>/move', methods=['POST'])
def move_summary(summary_id):
    data = request.json or {}
    category = data.get('category', '未分類')
    if data_manager.move_summary(summary_id, category):
        return jsonify({'success': True})
    return jsonify({'error': '移動失敗'}), 400

@app.route('/api/categories', methods=['GET'])
def get_categories():
    return jsonify({
        "categories": data_manager.categories,
        "groups": data_manager.config.get("category_groups", {}),
        "collapsed_groups": data_manager.config.get("collapsed_groups", [])
    })

@app.route('/api/categories', methods=['POST'])
def create_category():
    data = request.json or {}
    name = (data.get('name') or '').strip()
    if name and data_manager.add_category(name):
        return jsonify({'success': True})
    return jsonify({'error': '無效的分類名稱'}), 400

@app.route('/api/categories/<name>', methods=['PUT'])
def rename_category(name):
    data = request.json or {}
    new_name = (data.get('new_name') or '').strip()
    if new_name and data_manager.rename_category(name, new_name):
        return jsonify({'success': True})
    return jsonify({'error': '重新命名失敗'}), 400

@app.route('/api/categories/<name>', methods=['DELETE'])
def delete_category(name):
    if data_manager.delete_category(name):
        return jsonify({'success': True})
    return jsonify({'error': '刪除失敗'}), 400

@app.route('/api/category-groups', methods=['POST'])
def create_category_group():
    data = request.json or {}
    name = (data.get('name') or '').strip()
    if name and data_manager.add_category_group(name):
        return jsonify({'success': True})
    return jsonify({'error': '無效的分組名稱'}), 400

@app.route('/api/category-groups/<name>', methods=['DELETE'])
def delete_category_group(name):
    if data_manager.delete_category_group(name):
        return jsonify({'success': True})
    return jsonify({'error': '刪除失敗'}), 400

@app.route('/api/category-groups/<group_name>/add', methods=['POST'])
def add_category_to_group(group_name):
    data = request.json or {}
    category = (data.get('category') or '').strip()
    if category and data_manager.add_category_to_group(category, group_name):
        return jsonify({'success': True})
    return jsonify({'error': '添加失敗'}), 400

@app.route('/api/category-groups/<group_name>/remove', methods=['POST'])
def remove_category_from_group(group_name):
    data = request.json or {}
    category = (data.get('category') or '').strip()
    if category and data_manager.remove_category_from_group(category, group_name):
        return jsonify({'success': True})
    return jsonify({'error': '移除失敗'}), 400

@app.route('/api/category-groups/<group_name>/toggle', methods=['POST'])
def toggle_group_collapse(group_name):
    data_manager.toggle_group_collapse(group_name)
    return jsonify({'success': True})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        **data_manager.usage_stats,
        "records": [asdict(r) for r in data_manager.usage_records[:100]]
    })

@app.route('/api/stats/clear', methods=['POST'])
def clear_stats():
    data_manager.clear_usage_records()
    return jsonify({'success': True})

# =============================================================================
# HTML 模板
# =============================================================================

HTML_TEMPLATE = r'''
<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>🎬 字幕提取與整理工具 v3.0</title>
  <style>
    :root {
      --bg-primary: #f0f4f8;
      --bg-secondary: #ffffff;
      --bg-card: #ffffff;
      --text-primary: #1e293b;
      --text-secondary: #64748b;
      --accent: #3b82f6;
      --accent-green: #22c55e;
      --accent-yellow: #f59e0b;
      --accent-red: #ef4444;
      --border: #e2e8f0;
    }
    .dark-mode {
      --bg-primary: #1a1a2e;
      --bg-secondary: #16213e;
      --bg-card: rgba(255, 255, 255, 0.05);
      --text-primary: #e0e0e0;
      --text-secondary: #888;
      --accent: #00d9ff;
      --accent-green: #00ff88;
      --accent-yellow: #ffd700;
      --accent-red: #ff4444;
      --border: rgba(255, 255, 255, 0.1);
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg-primary);
      min-height: 100vh;
      color: var(--text-primary);
      transition: all 0.3s;
    }
    .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
    header { display:flex; justify-content:space-between; align-items:center; padding: 15px 0; flex-wrap: wrap; gap: 15px; }
    .logo {
      font-size: 1.6rem; font-weight: 700;
      background: linear-gradient(90deg, var(--accent), var(--accent-green));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .header-controls { display:flex; gap: 12px; align-items: center; }
    .theme-toggle {
      background: var(--bg-card); border: 1px solid var(--border);
      border-radius: 25px; padding: 6px 14px; cursor: pointer;
      display: flex; align-items: center; gap: 6px; color: var(--text-primary);
      transition: all 0.2s; font-size: 0.9rem;
    }
    .theme-toggle:hover { border-color: var(--accent); }

    .tabs { display:flex; gap:5px; margin-bottom: 20px; border-bottom:2px solid var(--border); padding-bottom:8px; flex-wrap: wrap; }
    .tab {
      padding: 10px 18px; background: transparent; border: none;
      color: var(--text-secondary); font-size: 0.95rem; cursor: pointer;
      border-radius: 8px 8px 0 0; transition: all 0.2s;
      display:flex; align-items:center; gap:6px;
    }
    .tab:hover { color: var(--accent); background: rgba(59, 130, 246, 0.1); }
    .tab.active { color:white; background: var(--accent); border-bottom:2px solid var(--accent); margin-bottom:-2px; }

    .panel { display:none; animation: fadeIn .3s; }
    .panel.active { display:block; }
    @keyframes fadeIn { from { opacity:0; } to { opacity:1; } }

    .card {
      background: var(--bg-card); border-radius: 12px; padding: 20px;
      margin-bottom: 16px; border: 1px solid var(--border);
    }
    .card h2 { color: var(--accent); margin-bottom: 15px; font-size: 1.1rem; display:flex; align-items:center; gap:8px; }

    textarea {
      width: 100%; height: 120px; background: var(--bg-primary);
      border: 1px solid var(--border); border-radius: 10px; padding: 12px;
      color: var(--text-primary); font-size: 0.95rem; resize: vertical;
    }
    textarea:focus { outline: none; border-color: var(--accent); }

    .form-row { display:flex; gap:12px; margin-top: 12px; flex-wrap: wrap; align-items:flex-end; }
    .form-group { flex:1; min-width: 180px; }
    .form-group label { display:block; margin-bottom:6px; color: var(--text-secondary); font-size: 0.85rem; }
    select, input[type="text"], input[type="number"], input[type="password"] {
      width:100%; padding: 10px 14px; background: var(--bg-primary);
      border: 1px solid var(--border); border-radius: 8px;
      color: var(--text-primary); font-size: 0.95rem;
    }
    select:focus, input:focus { outline:none; border-color: var(--accent); }

    .btn {
      padding: 10px 20px; border: none; border-radius: 8px; font-size: 0.95rem;
      cursor: pointer; transition: all 0.2s; display:inline-flex;
      align-items:center; gap:6px; font-weight:500;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent), #2563eb);
      color:white; box-shadow: 0 2px 8px rgba(59,130,246,0.3);
    }
    .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59,130,246,0.4); }
    .btn-primary:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
    .btn-secondary { background: var(--bg-card); color: var(--text-primary); border: 1px solid var(--border); }
    .btn-secondary:hover { border-color: var(--accent); color: var(--accent); }
    .btn-danger { background: var(--accent-red); color: white; }
    .btn-small { padding: 6px 12px; font-size: 0.85rem; }
    .btn-icon {
      width: 32px; height: 32px; padding: 0; border-radius: 6px;
      display:flex; align-items:center; justify-content:center;
    }

    .job-list { display:flex; flex-direction: column; gap: 10px; }
    .job-item {
      background: var(--bg-primary); border-radius: 10px; padding: 14px 16px;
      display:flex; align-items:center; gap: 12px;
      border: 1px solid transparent; transition: all 0.2s;
    }
    .job-item:hover { border-color: var(--accent); }
    .job-status { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
    .status-queued { background: var(--text-secondary); }
    .status-downloading { background: var(--accent-yellow); animation: pulse 1s infinite; }
    .status-transcribing { background: var(--accent); animation: pulse 1s infinite; }
    .status-completed { background: var(--accent-green); }
    .status-error { background: var(--accent-red); }
    .status-cancelled { background: var(--accent-red); }
    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.4; } }
    .job-info { flex:1; min-width: 0; }
    .job-title { font-weight:500; margin-bottom: 4px; word-break: break-word; }
    .job-url { font-size: 0.8rem; color: var(--text-secondary); white-space: nowrap; overflow:hidden; text-overflow: ellipsis; }
    .job-meta { font-size: 0.75rem; color: var(--text-secondary); margin-top: 4px; display:flex; gap: 12px; flex-wrap: wrap; }
    .job-progress { width: 120px; height: 6px; background: var(--border); border-radius: 3px; overflow:hidden; }
    .job-progress-bar { height:100%; background: linear-gradient(90deg, var(--accent), var(--accent-green)); transition: width 0.3s; }
    .job-actions { display:flex; gap:6px; flex-wrap: wrap; }

    .summary-grid { display:grid; grid-template-columns: 260px 1fr; gap: 20px; }
    @media (max-width: 900px) { .summary-grid { grid-template-columns: 1fr; } }
    .category-sidebar { position: sticky; top: 20px; }
    .category-list-container { background: var(--bg-card); border-radius: 12px; padding: 16px; border: 1px solid var(--border); max-height: 80vh; overflow-y: auto; }
    .category-header { display:flex; justify-content:space-between; align-items:center; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
    .category-header h3 { color: var(--accent); font-size: 0.95rem; display:flex; align-items:center; gap:6px; }

    .group-header {
      display:flex; align-items:center; gap:8px;
      padding: 8px 12px; margin: 8px 0 4px 0;
      background: rgba(59,130,246,0.1); border-radius: 8px;
      cursor: pointer; font-weight: 600; font-size: 0.9rem;
    }
    .group-header:hover { background: rgba(59,130,246,0.2); }
    .group-header .arrow { transition: transform 0.2s; }
    .group-header.collapsed .arrow { transform: rotate(-90deg); }
    .group-content { margin-left: 8px; }
    .group-content.collapsed { display: none; }

    .category-item {
      display:flex; align-items:center; gap:8px;
      padding: 10px 12px; border-radius: 8px; cursor:pointer;
      margin-bottom: 4px; transition: all 0.2s;
    }
    .category-item:hover { background: rgba(59,130,246,0.1); }
    .category-item.active { background: var(--accent); color: white; }
    .category-item.active .category-count { background: rgba(255,255,255,0.2); color: white; }
    .category-name { flex:1; white-space: nowrap; overflow:hidden; text-overflow: ellipsis; font-size: 0.9rem; }
    .category-count { background: var(--border); padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; min-width: 24px; text-align:center; flex-shrink:0; }

    .batch-actions {
      display: flex; gap: 10px; margin-bottom: 12px; padding: 12px;
      background: rgba(59,130,246,0.1); border-radius: 8px;
      align-items: center; flex-wrap: wrap;
    }
    .batch-actions.hidden { display: none; }
    .batch-actions span { color: var(--accent); font-weight: 500; }

    .summary-list { display:flex; flex-direction: column; gap: 12px; }
    .summary-item {
      background: var(--bg-card); border-radius: 12px; padding: 16px;
      border: 1px solid var(--border); transition: all 0.2s; position: relative;
    }
    .summary-item:hover { border-color: var(--accent); transform: translateY(-2px); }
    .summary-item.pinned { border-color: var(--accent-yellow); background: rgba(245,158,11,0.05); }
    .summary-item.pinned::before { content:'📌'; position:absolute; top: -6px; right: 12px; font-size: 1rem; }
    .summary-item.selected { border-color: var(--accent); background: rgba(59,130,246,0.1); }
    .summary-header { display:flex; justify-content: space-between; align-items:flex-start; gap: 12px; margin-bottom: 10px; }
    .summary-title {
      font-size: 1.05rem; font-weight: 600; color: var(--accent); cursor:pointer; flex: 1;
      word-break: break-word; line-height: 1.35;
    }
    .summary-title:hover { text-decoration: underline; }
    .summary-actions { display:flex; gap:4px; flex-shrink:0; align-items: center; }
    .summary-checkbox { width: 18px; height: 18px; cursor: pointer; }
    .summary-preview {
      color: var(--text-secondary); font-size: 0.9rem; line-height: 1.5;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow:hidden;
      margin-bottom: 10px;
    }
    .summary-meta { display:flex; gap: 12px; font-size: 0.8rem; color: var(--text-secondary); flex-wrap: wrap; }
    .summary-meta span { display:flex; align-items:center; gap:4px; }

    .stats-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; }
    .stat-card { background: var(--bg-card); border-radius: 12px; padding: 20px; text-align:center; border: 1px solid var(--border); }
    .stat-icon { font-size: 1.8rem; margin-bottom: 8px; }
    .stat-value {
      font-size: 1.6rem; font-weight:700;
      background: linear-gradient(90deg, var(--accent), var(--accent-green));
      -webkit-background-clip:text; -webkit-text-fill-color: transparent;
    }
    .stat-label { color: var(--text-secondary); margin-top: 6px; font-size: 0.85rem; }

    .modal {
      display:none; position: fixed; top:0; left:0; width:100%; height:100%;
      background: rgba(0,0,0,0.7); z-index: 1000;
      align-items:center; justify-content:center; padding: 20px;
    }
    .modal.active { display:flex; }
    .modal-content {
      background: var(--bg-secondary); border-radius: 16px; padding: 24px;
      max-width: 900px; max-height: 85vh; width: 100%; overflow-y:auto;
      border: 1px solid var(--border);
    }
    .modal-header {
      display:flex; justify-content: space-between; align-items:center;
      margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--border);
    }
    .modal-header h2 { color: var(--accent); font-size: 1.15rem; flex: 1; word-break: break-word; }
    .modal-close { background:none; border:none; color: var(--text-secondary); font-size: 1.8rem; cursor:pointer; margin-left: 10px; }
    .modal-close:hover { color: var(--accent-red); }

    .content-display {
      background: var(--bg-primary); border-radius: 10px; padding: 16px;
      max-height: 50vh; overflow-y:auto; line-height: 1.8;
    }
    .content-display a { color: var(--accent); text-decoration:none; }
    .content-display a:hover { text-decoration: underline; }
    .content-display h3 {
      font-size: 1.15rem; font-weight: 600; color: var(--accent);
      margin: 16px 0 8px 0; padding-bottom: 6px; border-bottom: 1px solid var(--border);
    }
    .content-display strong { color: var(--text-primary); }

    .action-bar {
      display:flex; gap: 10px; margin-top: 16px; flex-wrap: wrap;
      padding-top: 12px; border-top: 1px solid var(--border);
    }

    .video-info {
      background: rgba(59, 130, 246, 0.08); border-radius: 8px; padding: 12px;
      margin-bottom: 12px; font-size: 0.85rem;
    }
    .video-info-row { display:flex; gap: 16px; flex-wrap: wrap; }
    .video-info-item { display:flex; align-items:center; gap: 6px; }

    .empty-state { text-align:center; padding: 50px 20px; color: var(--text-secondary); }
    .empty-state .empty-icon { font-size: 3rem; margin-bottom: 15px; opacity: 0.5; }
    .data-table { width:100%; border-collapse: collapse; margin-top: 12px; }
    .data-table th, .data-table td { padding: 10px 12px; text-align:left; border-bottom: 1px solid var(--border); }
    .data-table th { color: var(--text-secondary); font-weight: 500; font-size: 0.85rem; }
    .input-group { display:flex; gap: 8px; }
    .input-group input { flex: 1; }

    .settings-section { margin-bottom: 20px; }
    .settings-section h3 { color: var(--accent); font-size: 1rem; margin-bottom: 12px; }

    .confirm-dialog {
      background: var(--bg-card); border-radius: 12px; padding: 20px;
      border: 1px solid var(--accent-yellow); margin-bottom: 16px;
    }
    .confirm-item {
      display: flex; justify-content: space-between; align-items: center;
      padding: 10px; background: var(--bg-primary); border-radius: 8px; margin: 8px 0;
    }
    .confirm-item-info { flex: 1; }
    .confirm-item-title { font-weight: 500; }
    .confirm-item-duration { color: var(--accent-yellow); font-size: 0.9rem; }

    .records-list { max-height: 400px; overflow-y: auto; }
    .record-item {
      display: flex; justify-content: space-between; align-items: center;
      padding: 10px 12px; border-bottom: 1px solid var(--border);
      font-size: 0.85rem;
    }
    .record-item:hover { background: rgba(59,130,246,0.05); }
    .record-model { color: var(--accent); font-weight: 500; }
    .record-cost { color: var(--accent-green); font-weight: 600; }
    .record-time { color: var(--text-secondary); font-size: 0.8rem; }

    .checkbox-group { display: flex; align-items: center; gap: 10px; margin: 10px 0; }
    .checkbox-group input[type="checkbox"] { width: 18px; height: 18px; }
    .checkbox-group label { color: var(--text-primary); }

    .api-status { display: inline-flex; align-items: center; gap: 6px; font-size: 0.85rem; }
    .api-status.ok { color: var(--accent-green); }
    .api-status.missing { color: var(--accent-red); }

    @media (max-width: 768px) {
      .header-controls { width: 100%; justify-content: space-between; }
      .form-row { flex-direction: column; }
      .job-item { flex-wrap: wrap; }
      .modal-content { padding: 16px; }
    }

    /* Tooltip 樣式 */
    .tooltip {
      position: relative;
      display: inline-block;
      cursor: help;
      margin-left: 4px;
      color: var(--accent);
      font-size: 0.9rem;
    }
    .tooltip .tooltiptext {
      visibility: hidden;
      width: 220px;
      background-color: var(--bg-card);
      color: var(--text-primary);
      text-align: left;
      border: 1px solid var(--accent);
      border-radius: 6px;
      padding: 10px;
      position: absolute;
      z-index: 1000;
      bottom: 125%;
      left: 50%;
      margin-left: -110px;
      opacity: 0;
      transition: opacity 0.3s;
      font-size: 0.8rem;
      font-weight: normal;
      line-height: 1.4;
      box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .tooltip:hover .tooltiptext {
      visibility: visible;
      opacity: 1;
    }
  </style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo">🎬 字幕提取與整理工具 v3.0</div>
    <div class="header-controls">
      <button class="theme-toggle" onclick="toggleTheme()">
        <span id="themeIcon">☀️</span>
        <span id="themeText">日間</span>
      </button>
    </div>
  </header>

  <div class="tabs">
    <button class="tab active" onclick="switchTab('extract')">📥 提取字幕</button>
    <button class="tab" onclick="switchTab('downloads')">📋 下載管理</button>
    <button class="tab" onclick="switchTab('summaries')">📝 整理結果</button>
    <button class="tab" onclick="switchTab('stats')">📊 使用統計</button>
    <button class="tab" onclick="switchTab('settings')">⚙️ 設定</button>
  </div>

  <!-- 提取字幕頁 -->
  <div id="panel-extract" class="panel active">
    <div class="card">
      <h2>📎 貼上影片網址</h2>
      <textarea id="urls-input" placeholder="每行一個網址，支援 yt-dlp 可處理的所有網站（YouTube、Bilibili 等）"></textarea>
      <div class="checkbox-group">
        <input type="checkbox" id="download-video-check">
        <label for="download-video-check">📹 順便下載原影片</label>
      </div>
      <div class="form-row">
        <button class="btn btn-primary" onclick="startExtraction()">🚀 開始提取</button>
        <button class="btn btn-secondary" onclick="openBatchWindow()">📦 批次輸入視窗</button>
      </div>
      <div style="margin-top:10px; color: var(--text-secondary); font-size: 0.85rem;">
        ※ 若影片有字幕會直接抓字幕；沒字幕則根據設定處理<br>
        ※ 超過閾值的長影片會先詢問確認
      </div>
    </div>

    <!-- 長影片確認區 -->
    <div id="long-video-confirm" class="confirm-dialog" style="display:none;">
      <h2 style="color: var(--accent-yellow); margin-bottom: 12px;">⚠️ 以下影片超過時長閾值</h2>
      <div id="long-video-list"></div>
      <div class="action-bar">
        <button class="btn btn-primary" onclick="confirmLongVideos()">✅ 全部繼續</button>
        <button class="btn btn-secondary" onclick="skipLongVideos()">⏭️ 全部略過</button>
        <button class="btn btn-secondary" onclick="cancelLongVideos()">❌ 取消</button>
      </div>
    </div>

    <div class="card">
      <h2>⏳ 處理佇列</h2>
      <div id="job-queue" class="job-list">
        <div class="empty-state"><div class="empty-icon">📭</div><p>尚無任務</p></div>
      </div>
    </div>
  </div>

  <!-- 下載管理頁 -->
  <div id="panel-downloads" class="panel">
    <div class="card">
      <h2>📋 已完成/失敗</h2>
      <div id="completed-jobs" class="job-list">
        <div class="empty-state"><div class="empty-icon">📭</div><p>尚無已完成的下載</p></div>
      </div>
    </div>
  </div>

  <!-- 整理結果頁 -->
  <div id="panel-summaries" class="panel">
    <div class="summary-grid">
      <div class="category-sidebar">
        <div class="category-list-container">
          <div class="category-header">
            <h3>📁 分類管理</h3>
            <div style="display:flex; gap:4px;">
              <button class="btn btn-small btn-secondary" onclick="showAddGroupModal()" title="新增分組">📂+</button>
              <button class="btn btn-small btn-secondary" onclick="showAddCategoryModal()" title="新增分類">➕</button>
            </div>
          </div>
          <div id="category-list"></div>
        </div>
      </div>
      <div>
        <div id="batch-actions" class="batch-actions hidden">
          <input type="checkbox" id="select-all-checkbox" onchange="toggleSelectAll()" style="width:18px;height:18px;">
          <span id="selected-count">已選 0 項</span>
          <select id="batch-move-category"></select>
          <button class="btn btn-small btn-secondary" onclick="batchMove()">📂 批次移動</button>
          <button class="btn btn-small btn-danger" onclick="batchDelete()">🗑️ 批次刪除</button>
        </div>
        <div id="summary-list" class="summary-list">
          <div class="empty-state"><div class="empty-icon">📝</div><p>尚無整理結果</p></div>
        </div>
      </div>
    </div>
  </div>

  <!-- 使用統計頁 -->
  <div id="panel-stats" class="panel">
    <div class="card">
      <h2>📊 總覽統計</h2>
      <div id="stats-grid" class="stats-grid"></div>
      <div class="action-bar">
        <button class="btn btn-danger btn-small" onclick="clearStats()">🗑️ 清空所有紀錄</button>
      </div>
    </div>
    <div class="card">
      <h2>🤖 LLM 使用明細</h2>
      <table class="data-table" id="llm-stats">
        <thead><tr><th>模型</th><th>輸入 Tokens</th><th>輸出 Tokens</th><th>預估費用 (TWD)</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card">
      <h2>🎙️ STT 使用明細</h2>
      <table class="data-table" id="stt-stats">
        <thead><tr><th>模型</th><th>轉錄時長</th><th>預估費用 (TWD)</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card">
      <h2>📜 單次紀錄（最近 100 筆）</h2>
      <div id="records-list" class="records-list"></div>
    </div>
  </div>

  <!-- 設定頁 -->
  <div id="panel-settings" class="panel">
    <div class="card">
      <h2>🔑 API 金鑰設定</h2>
      <div class="settings-section">
        <div class="form-row">
          <div class="form-group">
            <label>Google Gemini API Key</label>
            <input type="password" id="gemini-key-input" placeholder="輸入 Gemini API Key">
          </div>
          <div class="form-group" style="flex:0; min-width:auto;">
            <label>&nbsp;</label>
            <span id="gemini-status" class="api-status"></span>
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label>Groq API Key（用於 STT）</label>
            <input type="password" id="groq-key-input" placeholder="輸入 Groq API Key">
          </div>
          <div class="form-group" style="flex:0; min-width:auto;">
            <label>&nbsp;</label>
            <span id="groq-status" class="api-status"></span>
          </div>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>🤖 預設模型</h2>
      <div class="form-row">
        <div class="form-group">
            <label>LLM 整理模型</label>
            <select id="default-llm-model">
              <option value="google/gemini-2.5-flash-preview-09-2025">gemini-2.5-flash-preview-09-2025</option>
              <option value="google/gemini-3-flash-preview">gemini-3-flash-preview</option>
              <option value="google/gemini-3-flash-preview:thinking">gemini-3-flash-preview:thinking</option>
              <option value="google/gemini-3-pro-preview">gemini-3-pro-preview</option>
            </select>
        </div>
        <div class="form-group">
          <label>STT 轉錄模型</label>
          <select id="default-stt-model">
            <option value="whisper-large-v3">whisper-large-v3</option>
            <option value="whisper-large-v3-turbo">whisper-large-v3-turbo</option>
          </select>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>🎬 無字幕時的處理方式</h2>
      <div class="form-row">
        <div class="form-group">
          <select id="no-subtitle-action">
            <option value="llm_direct">直接送入 LLM 整理（不做 STT 逐字）</option>
            <option value="stt">進行 STT 語音轉文字（需要 Groq Key）</option>
            <option value="audio_only">只下載音軌（不自動處理）</option>
          </select>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>🎵 音檔設定</h2>
      <div class="form-row">
        <div class="form-group">
          <label>音檔儲存格式</label>
          <select id="audio-format">
            <option value="m4a">M4A (AAC)</option>
            <option value="mp3">MP3</option>
            <option value="wav">WAV</option>
          </select>
        </div>
        <div class="form-group">
          <label>人聲加強程度</label>
          <select id="speech-enhance">
            <option value="off">關閉</option>
            <option value="light">輕度</option>
            <option value="medium">中度</option>
            <option value="strong">強力（預設）</option>
          </select>
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>音訊加速倍率</label>
          <input id="llm-audio-speed" type="number" step="0.1" min="1.0" max="2.0">
        </div>
        <div class="form-group">
          <label>靜音門檻 (dB)</label>
          <input id="silence-noise-db" type="number" step="1">
        </div>
        <div class="form-group">
          <label>靜音最短秒數</label>
          <input id="silence-min-duration" type="number" step="0.1" min="0.1">
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>音訊分段時長（分鐘） 
            <span class="tooltip">ⓘ<span class="tooltiptext">長音訊自動切割。處理第 2 段起會參考前段內容作為上下文，確保連貫性。</span></span>
          </label>
          <input id="audio-segment-minutes" type="number" step="1" min="0" max="60" placeholder="0 表示不分段">
          <small style="color: var(--text-secondary); font-size: 0.85rem;">超過此時長的音訊將自動切段處理（0 = 不分段）</small>
        </div>
      </div>
      <div class="checkbox-group">
        <input type="checkbox" id="enable-query-repeat">
        <label for="enable-query-repeat">啟用提詞重複 
          <span class="tooltip">ⓘ<span class="tooltiptext">將完整的查詢內容(提示詞+音訊)重複發送給 AI，大幅提升準確度，但會使 Token 消耗加倍。</span></span>
        </label>
      </div>
    </div>

    <div class="card">
      <h2>⚙️ 其他設定</h2>
      <div class="form-row">
        <div class="form-group">
          <label>長影片閾值（分鐘）</label>
          <input id="long-video-threshold" type="number" step="1" min="1">
        </div>
      </div>
    </div>

    <div class="action-bar">
      <button class="btn btn-secondary" onclick="reloadConfig()">🔄 重新載入</button>
      <button class="btn btn-primary" onclick="saveConfig()">💾 儲存設定</button>
    </div>
  </div>
</div>

<!-- 字幕內容 Modal -->
<div id="content-modal" class="modal">
  <div class="modal-content">
    <div class="modal-header">
      <h2 id="modal-title">📄 字幕內容</h2>
      <button class="modal-close" onclick="closeModal('content-modal')">&times;</button>
    </div>
    <div id="modal-video-info" class="video-info" style="display:none;"></div>
    <div id="modal-content" class="content-display"></div>
    <div id="summarize-status-area" style="display:none; margin: 10px 0; padding: 12px; border-radius: 8px; background: rgba(var(--accent-rgb), 0.1); border: 1px solid var(--accent); color: var(--accent); text-align: center; font-weight: bold;">
      <div class="spinner-small" style="display:inline-block; margin-right:8px;"></div>
      <span id="summarize-status-text">🚀 Gemini 正在思考中...</span>
    </div>
    <div class="action-bar">
      <button id="summarize-btn" class="btn btn-primary" onclick="summarizeContent()" title="送交 Gemini 進行筆記整理與摘要">📝 AI 整理</button>
      <button id="cancel-summarize-btn" class="btn btn-danger" style="display:none;" onclick="cancelSummarize()" title="停止目前的 AI 整理請求">⛔ 中斷</button>
      <button class="btn btn-secondary" onclick="downloadSubtitle()" title="下載原始或經處理後的字幕檔案 (.vtt/.srt)">📄 字幕檔</button>
      <button class="btn btn-secondary" onclick="copyLlmInput()" title="複製提示詞與內容，手動提供給其他 AI 處理">📋 複製指令</button>
      <button class="btn btn-secondary" onclick="downloadAudio()" title="下載轉錄用的音訊檔">🎵 音檔</button>
      <button class="btn btn-secondary" onclick="downloadVideo()" title="下載原始影片檔">🎬 影片</button>
      <button class="btn btn-secondary" onclick="showManualImport()" title="手動貼入外部 AI 的整理結果">🧾 手動匯入</button>
    </div>
  </div>
</div>

<!-- 手動匯入 Modal -->
<div id="manual-modal" class="modal">
  <div class="modal-content">
    <div class="modal-header">
      <h2>🧾 手動匯入整理結果</h2>
      <button class="modal-close" onclick="closeModal('manual-modal')">&times;</button>
    </div>
    <div style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 10px;">
      把你在其他 LLM 跑出的結果貼在下面。支援格式：<br>
      1. 純整理文字（可包含【標題】xxx）<br>
      2. 格式化輸出（===字幕開始=== ... ===整理開始=== ...）
    </div>
    <textarea id="manual-output" placeholder="貼上整理結果..." style="height: 200px;"></textarea>
    <div class="action-bar">
      <input id="manual-model-name" type="text" style="flex:1; min-width: 220px;" placeholder="model_used（可留空）">
      <button class="btn btn-primary" onclick="importManual()">匯入</button>
    </div>
  </div>
</div>

<!-- 整理詳情 Modal -->
<div id="summary-modal" class="modal">
  <div class="modal-content">
    <div class="modal-header">
      <h2 id="summary-modal-title">📝 整理結果</h2>
      <button class="modal-close" onclick="closeModal('summary-modal')">&times;</button>
    </div>
    <div id="summary-video-info" class="video-info" style="display:none;"></div>
    <div id="summary-modal-content" class="content-display"></div>
    <div class="action-bar">
      <select id="move-category" title="更改此筆記的分類歸屬"></select>
      <button class="btn btn-secondary" onclick="moveSummary()" title="套用新的分類設定">📂 移動</button>
      <button class="btn btn-secondary" onclick="copySummary()" title="複製整篇筆記內容到剪貼簿">📋 複製</button>
      <button class="btn btn-secondary" onclick="openVideoLink()" title="在分頁開啟原始影片網址">🔗 開啟影片</button>
      <button class="btn btn-secondary" onclick="downloadSummaryVideo()" title="下載此筆記對應的原始影片">🎬 下載影片</button>
      <button class="btn btn-secondary" onclick="downloadSummarySubtitle()" title="下載此筆記對應的原始字幕">📄 下載字幕</button>
    </div>
  </div>
</div>

<!-- 新增分類 Modal -->
<div id="category-modal" class="modal">
  <div class="modal-content" style="max-width: 360px;">
    <div class="modal-header">
      <h2>➕ 新增分類</h2>
      <button class="modal-close" onclick="closeModal('category-modal')">&times;</button>
    </div>
    <div class="input-group">
      <input type="text" id="new-category-name" placeholder="分類名稱" onkeypress="if(event.key==='Enter')addCategory()">
      <button class="btn btn-primary" onclick="addCategory()">確定</button>
    </div>
  </div>
</div>

<!-- 新手教學 Modal -->
<div id="tutorial-modal" class="modal">
  <div class="modal-content" style="max-width: 500px;">
    <div class="modal-header">
      <h2>👋 歡迎使用字幕工具</h2>
      <button class="modal-close" onclick="closeModal('tutorial-modal')">&times;</button>
    </div>
    <div style="line-height: 1.6;">
      <p>這是您第一次開啟工具，讓我們簡單介紹一下流程：</p>
      <ol>
        <li><strong>📥 貼上網址</strong>：在「提取字幕」頁面貼上 YouTube 或 Bilibili 連結。</li>
        <li><strong>🚀 開始提取</strong>：程式會自動下載音軌並進行 STT 轉錄（如果沒字幕）。</li>
        <li><strong>📝 AI 整理</strong>：在「下載管理」點擊 <strong>👁️ 檢視</strong> 並按下 <strong>📝 AI 整理</strong>，讓 Gemini 產出筆記！</li>
        <li><strong>⚙️ 設定</strong>：記得先去「設定」填入您的 <strong>Gemini API Key</strong> 喔。</li>
      </ol>
      <p style="color: var(--accent-yellow);">💡 提示：長音與複雜內容可能需要幾分鐘整理，請耐心等候。</p>
    </div>
    <div class="action-bar">
      <button class="btn btn-primary" onclick="closeModal('tutorial-modal'); localStorage.setItem('tutorial_seen', 'true');" style="width: 100%;">我知道了！</button>
    </div>
  </div>
</div>

<!-- 指派分類到分組 Modal -->
<div id="assign-modal" class="modal">
  <div class="modal-content" style="max-width: 420px;">
    <div class="modal-header">
      <h2>📌 指派分類到分組</h2>
      <button class="modal-close" onclick="closeModal('assign-modal')">&times;</button>
    </div>

    <div class="form-row">
      <div class="form-group">
        <label>分類</label>
        <select id="assign-category"></select>
      </div>
      <div class="form-group">
        <label>分組</label>
        <select id="assign-group"></select>
      </div>
    </div>

    <div class="action-bar">
      <button class="btn btn-primary" onclick="assignCategoryToGroup()">確定加入</button>
    </div>
  </div>
</div>

<!-- 編輯標題 Modal -->
<div id="edit-title-modal" class="modal">
  <div class="modal-content" style="max-width: 450px;">
    <div class="modal-header">
      <h2>✏️ 編輯標題</h2>
      <button class="modal-close" onclick="closeModal('edit-title-modal')">&times;</button>
    </div>
    <input type="hidden" id="edit-title-id">
    <div class="input-group">
      <input type="text" id="edit-title-input" placeholder="新標題" onkeypress="if(event.key==='Enter')saveTitle()">
      <button class="btn btn-primary" onclick="saveTitle()">儲存</button>
    </div>
  </div>
</div>

<script>
  let jobs = {}, summaries = {}, categories = {}, categoryGroups = {}, collapsedGroups = [];
  let config = {};
  let currentJobId = null, currentSummaryId = null, currentCategory = '全部';
  let isDarkMode = false;
  let selectedSummaries = new Set();
  let pendingLongVideos = [];
  let pendingShortVideos = [];
  let currentSummarizeTaskId = null;

  document.addEventListener('DOMContentLoaded', async () => {
    await reloadConfig();
    await loadJobs();
    await loadSummaries();
    await loadCategories();
    await loadStats();

    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') toggleTheme();

    // 檢查是否初次顯示教學
    if (!localStorage.getItem('tutorial_seen')) {
      document.getElementById('tutorial-modal').classList.add('active');
    }

    setInterval(loadJobs, 2000);
  });

  function toggleTheme() {
    isDarkMode = !isDarkMode;
    document.body.classList.toggle('dark-mode', isDarkMode);
    document.getElementById('themeIcon').textContent = isDarkMode ? '🌙' : '☀️';
    document.getElementById('themeText').textContent = isDarkMode ? '夜間' : '日間';
    localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
  }

  function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelector(`[onclick="switchTab('${tabName}')"]`).classList.add('active');
    document.getElementById(`panel-${tabName}`).classList.add('active');
    if (tabName === 'stats') loadStats();
    if (tabName === 'summaries') { loadSummaries(); loadCategories(); }
  }

  async function reloadConfig() {
    try {
      const res = await fetch('/api/config');
      config = await res.json();
      
      // 更新 API 狀態顯示
      document.getElementById('gemini-status').innerHTML = config.has_gemini_key 
        ? '<span class="api-status ok">✅ 已設定</span>'
        : '<span class="api-status missing">❌ 未設定</span>';
      document.getElementById('groq-status').innerHTML = config.has_groq_key
        ? '<span class="api-status ok">✅ 已設定</span>'
        : '<span class="api-status missing">❌ 未設定</span>';

      // 填入設定值
      document.getElementById('default-llm-model').value = config.default_llm_model || 'google/gemini-3-flash-preview:thinking';
      document.getElementById('default-stt-model').value = config.default_stt_model || 'whisper-large-v3';
      document.getElementById('no-subtitle-action').value = config.no_subtitle_action || 'llm_direct';
      document.getElementById('audio-format').value = config.audio_format || 'm4a';
      document.getElementById('speech-enhance').value = config.speech_enhance_preset || 'strong';
      document.getElementById('llm-audio-speed').value = config.llm_audio_speed ?? 1.5;
      document.getElementById('silence-noise-db').value = config.silence_noise_db ?? -40;
      document.getElementById('silence-min-duration').value = config.silence_min_duration ?? 1.0;
      document.getElementById('audio-segment-minutes').value = config.audio_segment_minutes ?? 0;
      document.getElementById('enable-query-repeat').checked = config.enable_query_repeat || false;
      document.getElementById('long-video-threshold').value = config.long_video_threshold_minutes ?? 30;
      document.getElementById('download-video-check').checked = config.download_video || false;

      // 更新 STT 選項可用性
      updateSttOptionAvailability();
    } catch(e) {
      console.error(e);
    }
  }

  function updateSttOptionAvailability() {
    const sttOption = document.querySelector('#no-subtitle-action option[value="stt"]');
    if (sttOption) {
      if (!config.has_groq_key) {
        sttOption.disabled = true;
        sttOption.textContent = '進行 STT 語音轉文字（需要 Groq Key - 未設定）';
      } else {
        sttOption.disabled = false;
        sttOption.textContent = '進行 STT 語音轉文字（需要 Groq Key）';
      }
    }
  }

  async function saveConfig() {
    try {
      const data = {
        gemini_api_key: document.getElementById('gemini-key-input').value.trim(),
        groq_api_key: document.getElementById('groq-key-input').value.trim(),
        default_llm_model: document.getElementById('default-llm-model').value,
        default_stt_model: document.getElementById('default-stt-model').value,
        no_subtitle_action: document.getElementById('no-subtitle-action').value,
        audio_format: document.getElementById('audio-format').value,
        speech_enhance_preset: document.getElementById('speech-enhance').value,
        llm_audio_speed: parseFloat(document.getElementById('llm-audio-speed').value) || 1.5,
        silence_noise_db: parseFloat(document.getElementById('silence-noise-db').value) || -40,
        silence_min_duration: parseFloat(document.getElementById('silence-min-duration').value) || 1.0,
        audio_segment_minutes: parseInt(document.getElementById('audio-segment-minutes').value) || 0,
        enable_query_repeat: document.getElementById('enable-query-repeat').checked,
        long_video_threshold_minutes: parseFloat(document.getElementById('long-video-threshold').value) || 30,
        download_video: document.getElementById('download-video-check').checked,
      };

      // 只傳有值的 API Key
      if (!data.gemini_api_key) delete data.gemini_api_key;
      if (!data.groq_api_key) delete data.groq_api_key;

      const res = await fetch('/api/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      const result = await res.json();
      if (!result.success) { alert('儲存失敗'); return; }
      
      // 清空密碼輸入框
      document.getElementById('gemini-key-input').value = '';
      document.getElementById('groq-key-input').value = '';
      
      await reloadConfig();
      alert('✅ 已儲存');
    } catch(e) { alert('儲存失敗'); }
  }

  async function loadJobs() {
    try {
      const res = await fetch('/api/jobs');
      const data = await res.json();
      jobs = {}; data.forEach(j => jobs[j.id] = j);
      renderJobQueue(); renderCompletedJobs();
    } catch (e) { console.error(e); }
  }

  async function loadSummaries() {
    try {
      const res = await fetch('/api/summaries');
      const data = await res.json();
      summaries = {}; data.forEach(s => summaries[s.id] = s);
      renderSummaries();
    } catch (e) { console.error(e); }
  }

  async function loadCategories() {
    try {
      const res = await fetch('/api/categories');
      const data = await res.json();
      categories = data.categories || {};
      categoryGroups = data.groups || {};
      collapsedGroups = data.collapsed_groups || [];
      renderCategories();
    } catch (e) { console.error(e); }
  }

  function showAssignModal(categoryName = '') {
    // 填分類選單
    const catSel = document.getElementById('assign-category');
    const allCats = Object.keys(categories || {});
    catSel.innerHTML = allCats.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('');
    if (categoryName) catSel.value = categoryName;
  
    // 填分組選單
    const groupSel = document.getElementById('assign-group');
    const groups = Object.keys(categoryGroups || {});
    groupSel.innerHTML = groups.map(g => `<option value="${escapeHtml(g)}">${escapeHtml(g)}</option>`).join('');
  
    if (groups.length === 0) {
      alert('請先新增分組');
      return;
    }

    document.getElementById('assign-modal').classList.add('active');
  }

  async function assignCategoryToGroup() {
    const category = document.getElementById('assign-category').value;
    const group = document.getElementById('assign-group').value;
    if (!category || !group) return;
  
    await fetch(`/api/category-groups/${encodeURIComponent(group)}/add`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category })
    });

    closeModal('assign-modal');
    await loadCategories();  // 會重畫分組/分類列表
  }

  async function removeCategoryFromGroup(category, group) {
    await fetch(`/api/category-groups/${encodeURIComponent(group)}/remove`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category })
    });
    await loadCategories();
  }

  async function loadStats() {
    try {
      const res = await fetch('/api/stats');
      const stats = await res.json();
      renderStats(stats);
    } catch (e) { console.error(e); }
  }

  function openBatchWindow() {
    window.open('/batch', '_blank', 'width=1200,height=900');
  }

  async function startExtraction() {
    const input = document.getElementById('urls-input').value;
    const urls = input.split('\n').filter(u => u.trim());
    if (urls.length === 0) { alert('請輸入網址'); return; }

    // 更新下載影片設定
    const downloadVideo = document.getElementById('download-video-check').checked;
    await fetch('/api/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ download_video: downloadVideo })
    });

    // 檢查時長
    const checkRes = await fetch('/api/jobs/check-duration', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ urls })
    });
    const checkData = await checkRes.json();

    pendingLongVideos = checkData.filter(v => v.needs_confirmation);
    pendingShortVideos = checkData.filter(v => !v.needs_confirmation);

    if (pendingLongVideos.length > 0) {
      showLongVideoConfirm();
    } else {
      await submitJobs(urls);
    }
  }

  function showLongVideoConfirm() {
    const container = document.getElementById('long-video-list');
    container.innerHTML = pendingLongVideos.map(v => `
      <div class="confirm-item">
        <div class="confirm-item-info">
          <div class="confirm-item-title">${escapeHtml(v.title)}</div>
          <div class="confirm-item-duration">⏱️ ${v.duration_str}</div>
        </div>
        <div>
          <button class="btn btn-small btn-secondary" onclick="skipSingleLongVideo('${escapeJs(v.url)}')">略過</button>
        </div>
      </div>
    `).join('');
    document.getElementById('long-video-confirm').style.display = 'block';
  }

  function skipSingleLongVideo(url) {
    pendingLongVideos = pendingLongVideos.filter(v => v.url !== url);
    if (pendingLongVideos.length === 0) {
      document.getElementById('long-video-confirm').style.display = 'none';
      if (pendingShortVideos.length > 0) {
        submitJobs(pendingShortVideos.map(v => v.url));
      }
    } else {
      showLongVideoConfirm();
    }
  }

  async function confirmLongVideos() {
    document.getElementById('long-video-confirm').style.display = 'none';
    const allUrls = [...pendingShortVideos.map(v => v.url), ...pendingLongVideos.map(v => v.url)];
    await submitJobs(allUrls);
  }

  function skipLongVideos() {
    document.getElementById('long-video-confirm').style.display = 'none';
    if (pendingShortVideos.length > 0) {
      submitJobs(pendingShortVideos.map(v => v.url));
    }
    pendingLongVideos = [];
  }

  function cancelLongVideos() {
    document.getElementById('long-video-confirm').style.display = 'none';
    pendingLongVideos = [];
    pendingShortVideos = [];
  }

  async function submitJobs(urls) {
    try {
      const res = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ urls })
      });
      const newJobs = await res.json();
      newJobs.forEach(j => jobs[j.id] = j);
      document.getElementById('urls-input').value = '';
      renderJobQueue();
    } catch (e) { alert('建立任務失敗'); }
  }

  function escapeHtml(s) { return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
  function escapeJs(s) { return (s||'').replace(/\\/g,'\\\\').replace(/'/g,"\\'"); }

  function formatDate(dateStr) {
    if (dateStr && dateStr.length === 8) return `${dateStr.slice(0,4)}/${dateStr.slice(4,6)}/${dateStr.slice(6)}`;
    return dateStr || '';
  }

  function renderJobQueue() {
    const container = document.getElementById('job-queue');
    const pending = Object.values(jobs)
      .filter(j => ['queued', 'downloading', 'transcribing'].includes(j.status))
      .sort((a,b) => new Date(b.created_at) - new Date(a.created_at));

    if (pending.length === 0) {
      container.innerHTML = '<div class="empty-state"><div class="empty-icon">📭</div><p>尚無任務</p></div>';
      return;
    }

    container.innerHTML = pending.map(job => `
      <div class="job-item">
        <div class="job-status status-${job.status}"></div>
        <div class="job-info">
          <div class="job-title">${escapeHtml(job.title || '載入中...')}</div>
          <div class="job-url">${escapeHtml(job.url)}</div>
          <div class="job-meta">
            ${job.channel ? `<span>📺 ${escapeHtml(job.channel)}</span>` : ''}
            <span>🔧 ${escapeHtml(job.stage || '')}</span>
          </div>
        </div>
        <div class="job-progress"><div class="job-progress-bar" style="width:${job.progress || 0}%"></div></div>
        <span style="color: var(--text-secondary); font-size: 0.85rem;">${job.progress || 0}%</span>
        <div class="job-actions">
          <button class="btn btn-icon btn-secondary" onclick="cancelJob('${job.id}')" title="停止提取流程">⛔</button>
          <button class="btn btn-icon btn-secondary" onclick="deleteJob('${job.id}')" title="刪除任務紀錄">🗑️</button>
        </div>
      </div>
    `).join('');
  }

  function renderCompletedJobs() {
    const container = document.getElementById('completed-jobs');
    const completed = Object.values(jobs)
      .filter(j => ['completed','error','cancelled'].includes(j.status))
      .sort((a,b) => new Date(b.created_at) - new Date(a.created_at));

    if (completed.length === 0) {
      container.innerHTML = '<div class="empty-state"><div class="empty-icon">📭</div><p>尚無已完成的下載</p></div>';
      return;
    }

    container.innerHTML = completed.map(job => {
      const isSummarizing = job.is_summarizing;
      const hasGeminiKey = config.has_gemini_key;
      const canSummarize = hasGeminiKey && !isSummarizing;
      const btnText = isSummarizing ? '⏳ 整理中...' : (hasGeminiKey ? '📝 AI 整理' : '🚫 不可用');
      const btnClass = canSummarize ? 'btn-primary' : 'btn-secondary';
      const btnDisabled = canSummarize ? '' : 'disabled';

      return `
        <div class="job-item">
          <div class="job-status status-${job.status}"></div>
          <div class="job-info">
            <div class="job-title">${escapeHtml(job.title || '未知標題')}</div>
            <div class="job-url">${escapeHtml(job.url)}</div>
            <div class="job-meta">
              ${job.channel ? `<span>📺 ${escapeHtml(job.channel)}</span>` : ''}
              ${job.upload_date ? `<span>📅 ${formatDate(job.upload_date)}</span>` : ''}
              <span>🔧 ${escapeHtml(job.stage || '')}</span>
            </div>
            ${job.error_message ? `<div style="color: var(--accent-red); font-size: 0.8rem; margin-top: 4px;">❌ ${escapeHtml(job.error_message)}</div>` : ''}
          </div>
          <div class="job-actions">
            ${job.status === 'completed' ? `
              <button class="btn btn-secondary btn-small" onclick="showContent('${job.id}')" title="查看字幕內容、下載音檔或進行 AI 整理">👁️ 檢視</button>
              <button class="btn ${btnClass} btn-small" ${btnDisabled} onclick="summarizeJob('${job.id}', this)" title="送交 Gemini 進行筆記整理與摘要">
                ${btnText}
              </button>
              <button class="btn btn-secondary btn-small" onclick="copyLlmInputForJob('${job.id}')" title="複製完整提示詞與內容，手動貼給其他 AI">📋 指令</button>
              ${job.audio_path ? `<button class="btn btn-secondary btn-small" onclick="downloadAudioForJob('${job.id}')" title="下載處理過的 M4A 音檔">🎵</button>` : ''}
              ${job.video_path ? `<button class="btn btn-secondary btn-small" onclick="downloadVideoForJob('${job.id}')" title="下載原始影片檔案">🎬</button>` : ''}
              <button class="btn btn-secondary btn-small" onclick="showManualImportForJob('${job.id}')" title="手動貼上外部 AI 整理的結果">🧾</button>
            ` : ''}
            <button class="btn btn-icon btn-secondary" onclick="deleteJob('${job.id}')" title="永久刪除此任務與相關檔案">🗑️</button>
          </div>
        </div>
      `;
    }).join('');
  }

  async function cancelJob(jobId) {
    await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
    await loadJobs();
  }

  async function deleteJob(jobId) {
    if (!confirm('確定刪除？')) return;
    await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
    delete jobs[jobId];
    renderJobQueue(); renderCompletedJobs();
  }

  async function copyTextSmart(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch(e) {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.top = '-9999px';
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      return ok;
    }
  }

  async function copyLlmInputForJob(jobId) {
    try {
      const res = await fetch(`/api/jobs/${jobId}/llm-input`);
      const data = await res.json();
      if (data.error) { alert('❌ ' + data.error); return; }
      const ok = await copyTextSmart(data.content);
      alert(ok ? '✅ 已複製指令' : '複製失敗');
    } catch(e) { alert('複製失敗'); }
  }

  function downloadAudioForJob(jobId) {
    window.open(`/api/jobs/${jobId}/audio`, '_blank');
  }

  function downloadVideoForJob(jobId) {
    window.open(`/api/jobs/${jobId}/video`, '_blank');
  }

  function showManualImportForJob(jobId) {
    currentJobId = jobId;
    document.getElementById('manual-output').value = '';
    document.getElementById('manual-model-name').value = '';
    document.getElementById('manual-modal').classList.add('active');
  }

  async function summarizeJob(jobId, btn = null) {
    if (!config.has_gemini_key) {
      alert('未設定 Gemini API Key');
      return;
    }

    const b = btn;
    const originalText = b ? b.innerHTML : '';
    const taskId = 'task_' + Date.now();

    if (b) {
      b.disabled = true;
      // 動態更新按鈕文字提示
      const waitMessages = [
        '⏳ 處理中...',
        '⏳ 等待 Gemini...',
        '⏳ 請耐心等候...',
        '⏳ 整理筆記中...',
        '⏳ 即將完成...'
      ];
      let msgIdx = 0;
      b.innerHTML = waitMessages[0];
      const waitInterval = setInterval(() => {
        msgIdx = (msgIdx + 1) % waitMessages.length;
        b.innerHTML = waitMessages[msgIdx];
      }, 10000);
      b.dataset.waitInterval = waitInterval;
    }

    try {
      const res = await fetch('/api/summaries', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId, task_id: taskId })
      });
      const summary = await res.json();
      if (summary.cancelled) { alert('已取消'); return; }
      if (summary.error) { alert('❌ ' + summary.error); return; }
      summaries[summary.id] = summary;
      await loadCategories();
      await loadSummaries();
      loadStats();
      switchTab('summaries');
      showSummary(summary.id);
    } catch(e) {
      alert('整理失敗');
    } finally {
      if (b) {
        clearInterval(b.dataset.waitInterval);
        b.disabled = false;
        b.innerHTML = originalText;
      }
      await loadJobs(); // 刷新以更新整理狀態
    }
  }

  function renderCategories() {
    const container = document.getElementById('category-list');
    const allCount = Object.keys(summaries).length;

    // 找出哪些分類已在分組中
    const categoriesInGroups = new Set();
    for (const cats of Object.values(categoryGroups)) {
      cats.forEach(c => categoriesInGroups.add(c));
    }

    let html = `
      <div class="category-item ${currentCategory === '全部' ? 'active' : ''}" onclick="selectCategory('全部')">
        <span>📋</span><span class="category-name">全部</span><span class="category-count">${allCount}</span>
      </div>
    `;

    // 渲染分組
    for (const [groupName, groupCats] of Object.entries(categoryGroups)) {
      const isCollapsed = collapsedGroups.includes(groupName);
      const groupCount = groupCats.reduce((sum, cat) => sum + (categories[cat]?.length || 0), 0);
      
      html += `
        <div class="group-header ${isCollapsed ? 'collapsed' : ''}" onclick="toggleGroupCollapse('${escapeJs(groupName)}')">
          <span class="arrow">▼</span>
          <span style="flex:1;">${escapeHtml(groupName)}</span>
          <span class="category-count">${groupCount}</span>
          <button class="btn btn-icon btn-secondary" onclick="event.stopPropagation(); deleteGroup('${escapeJs(groupName)}')" title="刪除分組" style="width:24px;height:24px;">✖</button>
        </div>
        <div class="group-content ${isCollapsed ? 'collapsed' : ''}">
      `;
      
      for (const catName of groupCats) {
        const catIds = categories[catName] || [];
        html += `
          <div class="category-item ${currentCategory === catName ? 'active' : ''}" onclick="selectCategory('${escapeJs(catName)}')">
            <span>📂</span>
            <span class="category-name">${escapeHtml(catName)}</span>
            <span class="category-count">${catIds.length}</span>
            <button class="btn btn-icon btn-secondary"
              onclick="event.stopPropagation(); removeCategoryFromGroup('${escapeJs(catName)}','${escapeJs(groupName)}')"
              title="從分組移除"
              style="width:24px;height:24px;margin-left:auto;">
              ➖
            </button>
          </div>
        `;
      }
      
      html += '</div>';
    }

    // 渲染未分組的分類
    for (const [name, ids] of Object.entries(categories)) {
      if (categoriesInGroups.has(name)) continue;
      html += `
        <div class="category-item ${currentCategory === name ? 'active' : ''}" onclick="selectCategory('${escapeJs(name)}')">
          <span>${name === '未分類' ? '📁' : '📂'}</span><span class="category-name">${escapeHtml(name)}</span><span class="category-count">${ids.length}</span>
          ${name !== '未分類' ? `
          <button class="btn btn-icon btn-secondary"
            onclick="event.stopPropagation(); showAssignModal('${escapeJs(name)}')"
            title="加入分組"
            style="width:24px;height:24px;margin-left:auto;">
            ➕
          </button>` : ''}
        </div>
      `;
    }

    container.innerHTML = html;

    // 更新移動選單
    const moveSelect = document.getElementById('move-category');
    const batchMoveSelect = document.getElementById('batch-move-category');
    const opts = Object.keys(categories).map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('');
    if (moveSelect) moveSelect.innerHTML = opts;
    if (batchMoveSelect) batchMoveSelect.innerHTML = opts;
  }

  async function toggleGroupCollapse(groupName) {
    await fetch(`/api/category-groups/${encodeURIComponent(groupName)}/toggle`, { method: 'POST' });
    if (collapsedGroups.includes(groupName)) {
      collapsedGroups = collapsedGroups.filter(g => g !== groupName);
    } else {
      collapsedGroups.push(groupName);
    }
    renderCategories();
  }

  async function deleteGroup(groupName) {
    if (!confirm(`確定刪除分組「${groupName}」？分類本身不會被刪除。`)) return;
    await fetch(`/api/category-groups/${encodeURIComponent(groupName)}`, { method: 'DELETE' });
    await loadCategories();
  }

  function selectCategory(name) {
    currentCategory = name;
    selectedSummaries.clear();
    renderCategories();
    renderSummaries();
  }

  function renderSummaries() {
    const container = document.getElementById('summary-list');
    let items = Object.values(summaries);
    if (currentCategory !== '全部') items = items.filter(s => s.category === currentCategory);
    items.sort((a, b) => {
      if (a.pinned && !b.pinned) return -1;
      if (!a.pinned && b.pinned) return 1;
      return new Date(b.created_at) - new Date(a.created_at);
    });

    const batchBar = document.getElementById('batch-actions');
    if (items.length > 0) {
      batchBar.classList.remove('hidden');
    } else {
      batchBar.classList.add('hidden');
    }
    updateSelectedCount();

    if (items.length === 0) {
      container.innerHTML = '<div class="empty-state"><div class="empty-icon">📝</div><p>尚無整理結果</p></div>';
      return;
    }

    container.innerHTML = items.map(s => {
      let preview = (s.content || '').substring(0, 200).replace(/###\s*/g, '').replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      const modelName = (s.model_used || '').split('/').pop();
      const isSelected = selectedSummaries.has(s.id);
      return `
        <div class="summary-item ${s.pinned ? 'pinned' : ''} ${isSelected ? 'selected' : ''}">
          <div class="summary-header">
            <div class="summary-actions" style="margin-right: 8px;">
              <input type="checkbox" class="summary-checkbox" ${isSelected ? 'checked' : ''} onchange="toggleSelect('${s.id}')">
            </div>
            <div class="summary-title" onclick="showSummary('${s.id}')">${escapeHtml(s.title)}</div>
            <div class="summary-actions">
              <button class="btn btn-icon btn-secondary" onclick="togglePin('${s.id}')" title="${s.pinned ? '取消置頂' : '置頂'}">${s.pinned ? '📌' : '📍'}</button>
              <button class="btn btn-icon btn-secondary" onclick="showEditTitleModal('${s.id}')" title="編輯">✏️</button>
              <button class="btn btn-icon btn-secondary" onclick="deleteSummaryConfirm('${s.id}')" title="刪除">🗑️</button>
            </div>
          </div>
          <div class="summary-preview" onclick="showSummary('${s.id}')">${preview}...</div>
          <div class="summary-meta">
            <span>📁 ${escapeHtml(s.category)}</span>
            ${s.channel ? `<span>📺 ${escapeHtml(s.channel)}</span>` : ''}
            <span>🤖 ${escapeHtml(modelName || 'manual')}</span>
            ${s.upload_date ? `<span>📅 ${formatDate(s.upload_date)}</span>` : ''}
          </div>
        </div>
      `;
    }).join('');
  }

  function toggleSelect(id) {
    if (selectedSummaries.has(id)) selectedSummaries.delete(id);
    else selectedSummaries.add(id);
    renderSummaries();
  }

  function toggleSelectAll() {
    const checkbox = document.getElementById('select-all-checkbox');
    let items = Object.values(summaries);
    if (currentCategory !== '全部') items = items.filter(s => s.category === currentCategory);

    if (checkbox.checked) {
      items.forEach(s => selectedSummaries.add(s.id));
    } else {
      selectedSummaries.clear();
    }
    renderSummaries();
  }

  function updateSelectedCount() {
    document.getElementById('selected-count').textContent = `已選 ${selectedSummaries.size} 項`;
  }

  async function batchMove() {
    if (selectedSummaries.size === 0) { alert('請先選取項目'); return; }
    const category = document.getElementById('batch-move-category').value;
    await fetch('/api/summaries/batch-move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids: Array.from(selectedSummaries), category })
    });
    selectedSummaries.clear();
    await loadSummaries();
    await loadCategories();
  }

  async function batchDelete() {
    if (selectedSummaries.size === 0) { alert('請先選取項目'); return; }
    if (!confirm(`確定刪除 ${selectedSummaries.size} 項？`)) return;
    await fetch('/api/summaries/batch-delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids: Array.from(selectedSummaries) })
    });
    selectedSummaries.forEach(id => delete summaries[id]);
    selectedSummaries.clear();
    await loadCategories();
    renderSummaries();
  }

  async function togglePin(summaryId) {
    await fetch(`/api/summaries/${summaryId}/pin`, { method: 'POST' });
    await loadSummaries();
    await loadCategories();
  }

  async function deleteSummaryConfirm(summaryId) {
    if (!confirm('確定刪除？')) return;
    await fetch(`/api/summaries/${summaryId}`, { method: 'DELETE' });
    delete summaries[summaryId];
    await loadCategories();
    renderSummaries();
  }

  function formatSecondsToTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}時${m}分${s}秒`;
    if (m > 0) return `${m}分${s}秒`;
    return `${s}秒`;
  }

  function renderStats(stats) {
    document.getElementById('stats-grid').innerHTML = `
      <div class="stat-card"><div class="stat-icon">📥</div><div class="stat-value">${(stats.total_input_tokens || 0).toLocaleString()}</div><div class="stat-label">輸入 Tokens</div></div>
      <div class="stat-card"><div class="stat-icon">📤</div><div class="stat-value">${(stats.total_output_tokens || 0).toLocaleString()}</div><div class="stat-label">輸出 Tokens</div></div>
      <div class="stat-card"><div class="stat-icon">🎙️</div><div class="stat-value">${Math.round((stats.total_audio_seconds || 0) / 60)}</div><div class="stat-label">轉錄分鐘</div></div>
      <div class="stat-card"><div class="stat-icon">💰</div><div class="stat-value">NT$${(stats.total_cost_twd || 0).toFixed(2)}</div><div class="stat-label">預估總費用</div></div>
    `;

    const llmTbody = document.querySelector('#llm-stats tbody');
    let llmRows = '';
    for (const [model, data] of Object.entries(stats.by_model || {})) {
      llmRows += `<tr><td>${escapeHtml(model)}</td><td>${(data.input || 0).toLocaleString()}</td><td>${(data.output || 0).toLocaleString()}</td><td>NT$${(data.cost_twd || 0).toFixed(2)}</td></tr>`;
    }
    llmTbody.innerHTML = llmRows || '<tr><td colspan="4" style="text-align:center;color:var(--text-secondary);">無紀錄</td></tr>';

    const sttTbody = document.querySelector('#stt-stats tbody');
    let sttRows = '';
    for (const [model, data] of Object.entries(stats.stt_usage || {})) {
      sttRows += `<tr><td>${escapeHtml(model)}</td><td>${formatSecondsToTime(data.seconds || 0)}</td><td>NT$${(data.cost_twd || 0).toFixed(2)}</td></tr>`;
    }
    sttTbody.innerHTML = sttRows || '<tr><td colspan="3" style="text-align:center;color:var(--text-secondary);">無紀錄</td></tr>';

    const recordsList = document.getElementById('records-list');
    const records = stats.records || [];
    if (records.length === 0) {
      recordsList.innerHTML = '<div style="text-align:center; color: var(--text-secondary); padding: 20px;">無紀錄</div>';
    } else {
      recordsList.innerHTML = records.map(r => `
        <div class="record-item">
          <div>
            <span class="record-model">${escapeHtml(r.model)}</span>
            <span class="record-time">${new Date(r.timestamp).toLocaleString('zh-TW')}</span>
            <div style="font-size: 0.8rem; color: var(--text-secondary);">${escapeHtml(r.description || '')}</div>
          </div>
          <div class="record-cost">NT$${(r.cost_twd || 0).toFixed(4)}</div>
        </div>
      `).join('');
    }
  }

  async function clearStats() {
    if (!confirm('確定清空所有使用紀錄？')) return;
    await fetch('/api/stats/clear', { method: 'POST' });
    await loadStats();
    alert('✅ 已清空');
  }

  function showContent(jobId) {
    const job = jobs[jobId];
    if (!job) return;
    currentJobId = jobId;

    document.getElementById('modal-title').textContent = `📄 ${job.title || '字幕內容'}`;

    const infoDiv = document.getElementById('modal-video-info');
    infoDiv.style.display = 'block';
    infoDiv.innerHTML = `<div class="video-info-row">
      ${job.channel ? `<div class="video-info-item">📺 <strong>${escapeHtml(job.channel)}</strong></div>` : ''}
      ${job.upload_date ? `<div class="video-info-item">📅 ${formatDate(job.upload_date)}</div>` : ''}
      <div class="video-info-item">🔧 ${escapeHtml(job.stage || '')}</div>
    </div>`;

    let content = job.subtitle_with_time || job.subtitle_content || '（無字幕內容）';
    content = formatContent(content);
    document.getElementById('modal-content').innerHTML = content;
    document.getElementById('modal-content').scrollTop = 0;

    // 更新按鈕狀態
    const summarizeBtn = document.getElementById('summarize-btn');
    if (!config.has_gemini_key) {
      summarizeBtn.disabled = true;
      summarizeBtn.innerHTML = '🚫 不可用';
    } else if (job.is_summarizing) {
      summarizeBtn.disabled = true;
      summarizeBtn.innerHTML = '⏳ 整理中...';
    } else {
      summarizeBtn.disabled = false;
      summarizeBtn.innerHTML = '📝 AI 整理';
    }

    document.getElementById('content-modal').classList.add('active');
  }

  function formatContent(content) {
    content = content.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    content = content.replace(/^###\s*(.+)$/gm, '<h3>$1</h3>');
    content = content.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    content = content.replace(/\n/g, '<br>');
    return content;
  }

  function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
    if (modalId === 'content-modal') currentJobId = null;
    if (modalId === 'summary-modal') currentSummaryId = null;
  }

  async function summarizeContent() {
    if (!currentJobId) return;
    if (!config.has_gemini_key) {
      alert('未設定 Gemini API Key');
      return;
    }

    const btn = document.getElementById('summarize-btn');
    const cancelBtn = document.getElementById('cancel-summarize-btn');
    const statusArea = document.getElementById('summarize-status-area');
    const statusText = document.getElementById('summarize-status-text');
    const originalText = btn.innerHTML;

    currentSummarizeTaskId = 'task_' + Date.now();
    btn.disabled = true;
    statusArea.style.display = 'block';
    
    // 動態更新按鈕與提示區域文字
    const waitMessages = [
      '🚀 Gemini 正在思考中...',
      '⏳ 正在校對逐字稿並整理筆記...',
      '📚 內容較長，請耐心等候幾分鐘...',
      '🎨 正在美化排版與產出結果...',
      '🏁 快好了，請勿關閉視窗...'
    ];
    let msgIdx = 0;
    btn.innerHTML = '⏳ 處理中...';
    statusText.innerHTML = waitMessages[0];

    const waitInterval = setInterval(() => {
      msgIdx = (msgIdx + 1) % waitMessages.length;
      statusText.innerHTML = waitMessages[msgIdx];
    }, 10000);

    cancelBtn.style.display = 'inline-flex';

    try {
      const res = await fetch('/api/summaries', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: currentJobId, task_id: currentSummarizeTaskId })
      });
      const summary = await res.json();
      if (summary.cancelled) { alert('已取消'); return; }
      if (summary.error) { alert('❌ ' + summary.error); return; }
      summaries[summary.id] = summary;
      await loadCategories();
      renderSummaries();
      loadStats();
      closeModal('content-modal');
      switchTab('summaries');
      showSummary(summary.id);
    } catch(e) {
      alert('整理失敗');
    } finally {
      clearInterval(waitInterval);
      btn.disabled = false;
      btn.innerHTML = originalText;
      statusArea.style.display = 'none';
      cancelBtn.style.display = 'none';
      currentSummarizeTaskId = null;
      await loadJobs();
    }
  }

  async function cancelSummarize() {
    if (currentSummarizeTaskId) {
      await fetch(`/api/summaries/cancel/${currentSummarizeTaskId}`, { method: 'POST' });
    }
  }

  function downloadSubtitle() {
    if (currentJobId) window.open(`/api/jobs/${currentJobId}/subtitle`, '_blank');
  }

  function downloadAudio() {
    if (currentJobId) window.open(`/api/jobs/${currentJobId}/audio`, '_blank');
  }

  function downloadVideo() {
    if (currentJobId) window.open(`/api/jobs/${currentJobId}/video`, '_blank');
  }

  async function copyLlmInput() {
    if (!currentJobId) return;
    await copyLlmInputForJob(currentJobId);
  }

  function showManualImport() {
    if (!currentJobId) return;
    document.getElementById('manual-output').value = '';
    document.getElementById('manual-model-name').value = '';
    document.getElementById('manual-modal').classList.add('active');
  }

  async function importManual() {
    if (!currentJobId) return;
    const raw = (document.getElementById('manual-output').value || '').trim();
    if (!raw) { alert('請貼上整理結果'); return; }
    const model = (document.getElementById('manual-model-name').value || 'manual').trim();

    try {
      const res = await fetch('/api/summaries/manual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: currentJobId, raw_output: raw, model })
      });
      const summary = await res.json();
      if (summary.error) { alert('❌ ' + summary.error); return; }
      summaries[summary.id] = summary;
      await loadCategories();
      await loadSummaries();
      await loadJobs();
      closeModal('manual-modal');
      closeModal('content-modal');
      switchTab('summaries');
      showSummary(summary.id);
    } catch(e) {
      alert('匯入失敗');
    }
  }

  function showSummary(summaryId) {
    const summary = summaries[summaryId];
    if (!summary) return;
    currentSummaryId = summaryId;
    document.getElementById('summary-modal-title').textContent = `📝 ${summary.title}`;

    const infoDiv = document.getElementById('summary-video-info');
    infoDiv.style.display = 'block';
    infoDiv.innerHTML = `<div class="video-info-row">
      ${summary.video_title ? `<div class="video-info-item">🎬 ${escapeHtml(summary.video_title)}</div>` : ''}
      ${summary.channel ? `<div class="video-info-item">📺 ${escapeHtml(summary.channel)}</div>` : ''}
      ${summary.upload_date ? `<div class="video-info-item">📅 ${formatDate(summary.upload_date)}</div>` : ''}
    </div>`;

    let content = summary.content_with_time || summary.content || '';
    const contentDiv = document.getElementById('summary-modal-content');
    contentDiv.innerHTML = formatContent(content);
    contentDiv.scrollTop = 0;

    document.getElementById('move-category').value = summary.category;
    document.getElementById('summary-modal').classList.add('active');
  }

  async function moveSummary() {
    if (!currentSummaryId) return;
    const category = document.getElementById('move-category').value;
    await fetch(`/api/summaries/${currentSummaryId}/move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category })
    });
    await loadSummaries();
    await loadCategories();
    closeModal('summary-modal');
  }

  async function copySummary() {
    const summary = summaries[currentSummaryId];
    if (!summary) return;
    const ok = await copyTextSmart(summary.content || '');
    alert(ok ? '✅ 已複製' : '複製失敗');
  }

  function openVideoLink() {
    const summary = summaries[currentSummaryId];
    if (summary && summary.video_url) window.open(summary.video_url, '_blank');
  }

  function downloadSummaryVideo() {
    const summary = summaries[currentSummaryId];
    if (summary && summary.job_id) {
      window.open(`/api/jobs/${summary.job_id}/video`, '_blank');
    }
  }

  function downloadSummarySubtitle() {
    const summary = summaries[currentSummaryId];
    if (summary && summary.job_id) {
      window.open(`/api/jobs/${summary.job_id}/subtitle`, '_blank');
    }
  }

  function showAddCategoryModal() {
    document.getElementById('new-category-name').value = '';
    document.getElementById('category-modal').classList.add('active');
    document.getElementById('new-category-name').focus();
  }

  async function addCategory() {
    const name = (document.getElementById('new-category-name').value || '').trim();
    if (!name) return;
    await fetch('/api/categories', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    await loadCategories();
    closeModal('category-modal');
  }

  function showAddGroupModal() {
    document.getElementById('new-group-name').value = '';
    document.getElementById('group-modal').classList.add('active');
    document.getElementById('new-group-name').focus();
  }

  async function addGroup() {
    const name = (document.getElementById('new-group-name').value || '').trim();
    if (!name) return;
    await fetch('/api/category-groups', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    await loadCategories();
    closeModal('group-modal');
  }

  function showEditTitleModal(id) {
    const summary = summaries[id];
    if (!summary) return;
    document.getElementById('edit-title-id').value = id;
    document.getElementById('edit-title-input').value = summary.title;
    document.getElementById('edit-title-modal').classList.add('active');
    document.getElementById('edit-title-input').focus();
  }

  async function saveTitle() {
    const id = document.getElementById('edit-title-id').value;
    const newTitle = (document.getElementById('edit-title-input').value || '').trim();
    if (!newTitle) return;
    await fetch(`/api/summaries/${id}/title`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: newTitle })
    });
    summaries[id].title = newTitle;
    renderSummaries();
    closeModal('edit-title-modal');
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      document.querySelectorAll('.modal.active').forEach(m => m.classList.remove('active'));
      currentJobId = null;
      currentSummaryId = null;
    }
  });

  document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.classList.remove('active');
        currentJobId = null;
        currentSummaryId = null;
      }
    });
  });
  function showAssignModal(categoryName) {
    const catSelect = document.getElementById('assign-category');
    catSelect.innerHTML = Object.keys(categories)
      .filter(c => c !== '未分類')
      .map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`)
      .join('');
    
    if (categoryName) catSelect.value = categoryName;

    const groupSelect = document.getElementById('assign-group');
    groupSelect.innerHTML = Object.keys(categoryGroups)
      .map(g => `<option value="${escapeHtml(g)}">${escapeHtml(g)}</option>`)
      .join('');
      
    if (groupSelect.options.length === 0) {
      alert('請先建立分組');
      return;
    }

    document.getElementById('assign-modal').classList.add('active');
  }

  async function assignCategoryToGroup() {
     const category = document.getElementById('assign-category').value;
     const group = document.getElementById('assign-group').value;
     if(!category || !group) return;
     
     await fetch(`/api/category-groups/${encodeURIComponent(group)}/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category })
     });
     await loadCategories();
     closeModal('assign-modal');
  }

  async function removeCategoryFromGroup(category, group) {
    if (!confirm(`確定將「${category}」從「${group}」移除？`)) return;
    await fetch(`/api/category-groups/${encodeURIComponent(group)}/remove`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category })
    });
    await loadCategories();
  }

</script>
</body>
</html>
'''

BATCH_TEMPLATE = r'''
<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>批次提取與整理</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #e5e7eb;
      --primary: #2563eb;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: "Segoe UI", sans-serif; }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 20px; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px; margin-bottom: 16px; }
    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    .btn { border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 8px 12px; cursor: pointer; }
    .btn-primary { background: var(--primary); border-color: var(--primary); color: #fff; }
    .item { border: 1px solid var(--line); border-radius: 10px; padding: 12px; margin-top: 10px; }
    .item textarea { width: 100%; min-height: 120px; resize: vertical; margin-top: 8px; border: 1px solid var(--line); border-radius: 8px; padding: 10px; font-family: Consolas, monospace; }
    .item input[type=text] { width: 280px; max-width: 100%; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
    .hint { color: var(--muted); font-size: 13px; margin-top: 6px; }
    .num { width: 72px; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
    .grid { width: 100%; border-collapse: collapse; font-size: 14px; }
    .grid th, .grid td { border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }
    .tag { padding: 2px 8px; border-radius: 999px; font-size: 12px; display: inline-block; }
    .s-pending { background: #eef2ff; color: #4338ca; }
    .s-extracting { background: #e0f2fe; color: #0369a1; }
    .s-summarizing { background: #fef3c7; color: #92400e; }
    .s-completed { background: #dcfce7; color: #166534; }
    .s-error { background: #fee2e2; color: #991b1b; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h2 style="margin:0 0 8px 0;">批次提取 + AI 整理</h2>
      <div class="row" style="margin-bottom:10px;">
        <label>同時執行數</label>
        <input id="concurrency" class="num" type="number" min="1" max="3" value="3" />
        <span class="hint">預設 3，最大 3</span>
      </div>
      <div class="row">
        <button class="btn" onclick="addItem()">+ 新增項目</button>
        <button id="start-batch-btn" class="btn btn-primary" onclick="startBatch()">開始批次</button>
      </div>
      <div class="hint">每個項目可填不同分類名。分類會自動加入現有分類，若不存在則自動建立。</div>
      <div id="items"></div>
    </div>

    <div class="card">
      <div id="summary"></div>
      <table class="grid">
        <thead>
          <tr>
            <th style="width:140px;">狀態</th>
            <th style="width:180px;">分類</th>
            <th>網址</th>
            <th style="width:220px;">訊息</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </div>

  <script>
    let batchId = null;
    let pollTimer = null;
    let isStartingBatch = false;

    function itemHtml() {
      return `
      <div class="item">
        <div class="row">
          <input type="text" class="category" placeholder="分類名，例如：投資/語言學習/科技新聞" />
          <button class="btn" onclick="this.closest('.item').remove()">刪除項目</button>
        </div>
        <textarea class="urls" placeholder="每行一個網址"></textarea>
      </div>`;
    }

    function addItem() {
      const box = document.getElementById('items');
      box.insertAdjacentHTML('beforeend', itemHtml());
    }

    function normalizeStatus(status) {
      if (!status) return 'pending';
      return ['pending','extracting','summarizing','completed','error'].includes(status) ? status : 'error';
    }

    function statusText(status) {
      const map = {
        pending: '待處理',
        extracting: '提取中',
        summarizing: '整理中',
        completed: '完成',
        error: '失敗'
      };
      return map[status] || status;
    }

    function renderTask(task) {
      const c = task.counts || {};
      document.getElementById('summary').innerHTML = `
        <div class="row">
          <strong>批次 ${task.id}</strong>
          <span>狀態：${task.status}</span>
          <span>總數：${c.total || 0}</span>
          <span style="color:#0369a1;">提取中：${c.extracting || 0}</span>
          <span style="color:#92400e;">整理中：${c.summarizing || 0}</span>
          <span style="color:#166534;">完成：${c.completed || 0}</span>
          <span style="color:#991b1b;">失敗：${c.error || 0}</span>
        </div>`;

      const rows = (task.items || []).map(item => {
        const status = normalizeStatus(item.status);
        return `<tr>
          <td><span class="tag s-${status}">${statusText(status)}</span></td>
          <td>${escapeHtml(item.category || '未分類')}</td>
          <td style="word-break:break-all;">${escapeHtml(item.url || '')}</td>
          <td>${escapeHtml(item.message || '')}</td>
        </tr>`;
      }).join('');
      document.getElementById('rows').innerHTML = rows;
    }

    async function startBatch() {
      if (isStartingBatch) return;
      const startBtn = document.getElementById('start-batch-btn');
      const concurrency = Math.max(1, Math.min(3, parseInt(document.getElementById('concurrency').value || '3', 10)));
      document.getElementById('concurrency').value = String(concurrency);
      const items = Array.from(document.querySelectorAll('.item')).map(el => {
        const category = (el.querySelector('.category').value || '').trim();
        const urls = (el.querySelector('.urls').value || '').split(/\r?\n/).map(v => v.trim()).filter(Boolean);
        return { category, urls };
      }).filter(i => i.urls.length > 0);

      if (items.length === 0) {
        alert('請至少新增一個有網址的項目');
        return;
      }

      isStartingBatch = true;
      if (startBtn) {
        startBtn.disabled = true;
        startBtn.textContent = '送出中...';
      }
      try {
        const res = await fetch('/api/batch/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ concurrency, items })
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          alert(data.error || '批次啟動失敗');
          return;
        }

        batchId = data.batch_id;
        if (pollTimer) clearInterval(pollTimer);
        await pollBatch();
        pollTimer = setInterval(pollBatch, 2000);
      } finally {
        isStartingBatch = false;
        if (startBtn) {
          startBtn.disabled = false;
          startBtn.textContent = '開始批次';
        }
      }
    }

    async function pollBatch() {
      if (!batchId) return;
      const res = await fetch(`/api/batch/${batchId}`);
      const task = await res.json();
      if (!res.ok || task.error) return;
      renderTask(task);
      if (task.status === 'completed' && pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    }

    function escapeHtml(s) {
      return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    addItem();
  </script>
</body>
</html>
'''

# =============================================================================
# 主程式
# =============================================================================

def ensure_ytdlp():
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=5, env=env_without_proxy()
        )
        print(f"✅ yt-dlp: {result.stdout.strip()}")
    except:
        print("❌ yt-dlp 未安裝，正在安裝...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp", "-q"])

def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5)
        print("✅ ffmpeg 已可用")
    except:
        print("⚠️ 找不到 ffmpeg，部分功能可能無法使用")

def open_browser_later(url: str, delay: float = 1.0):
    def _open():
        try:
            webbrowser.open(url, new=2)
        except:
            pass
    threading.Timer(delay, _open).start()

def main():
    global APP_PORT
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    APP_PORT = args.port

    print("=" * 60)
    print("🎬 字幕提取與整理工具 v3.0")
    print("=" * 60)

    ensure_ytdlp()
    ensure_ffmpeg()

    # 顯示 API Key 狀態
    if data_manager.config.get("gemini_api_key"):
        print("✅ Gemini API Key 已設定")
    else:
        print("⚠️ 未設定 Gemini API Key（請在設定頁填入）")

    if data_manager.config.get("groq_api_key"):
        print("✅ Groq API Key 已設定")
    else:
        print("⚠️ 未設定 Groq API Key（STT 功能需要）")

    print(f"\n📁 資料目錄: {DATA_DIR.absolute()}")

    start_worker_pool()

    url = f"http://localhost:{args.port}"
    print(f"\n🚀 啟動中... {url}")
    print("Ctrl+C 停止")

    open_browser_later(url, delay=1.2)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)

if __name__ == "__main__":
    main()
