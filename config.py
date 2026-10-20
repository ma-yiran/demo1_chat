from pathlib import Path
from dotenv import load_dotenv

# 根目录 = 本文件所在目录（config.py 就在根目录）
BASE_DIR = Path(__file__).resolve().parent

# 载入根目录的 .env
load_dotenv(BASE_DIR / ".env")
