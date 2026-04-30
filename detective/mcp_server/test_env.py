#!/usr/bin/env python3
"""Test .env loading"""

import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / '.env'
print(f'Loading from: {env_path.absolute()}')
print(f'Exists: {env_path.exists()}')

if env_path.exists():
    load_dotenv(env_path)
    print('[OK] Loaded .env')
else:
    print('[WARN] .env not found')

print(f'ORS_API_KEY: {bool(os.getenv("ORS_API_KEY"))}')
print(f'GROQ_API_KEY: {bool(os.getenv("GROQ_API_KEY"))}')
print(f'GEMINI_API_KEY: {bool(os.getenv("GEMINI_API_KEY"))}')
