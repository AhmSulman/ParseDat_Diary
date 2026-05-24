[app]
title = MAAN
package.name = maan
package.domain = com.maan.chatbooks
source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,json,txt
source.exclude_dirs = .venv, .git, .buildozer, data/models, data/cache, data/checkpoints, data/txt, data/json, __pycache__, .github
source.exclude_patterns = *.pyc, *.pyo, *.gguf, *.bin, *.log

version = 1.0.0

requirements = hostpython3==3.12.13,python3==3.12,kivy==2.3.0,kivymd==1.2.0,requests,pillow,pymupdf

# Android orientation
orientation = portrait

# Fullscreen on Android
fullscreen = 0

# Supported APIs
android.minapi = 26
android.api = 33
android.ndk = 25b
android.archs = arm64-v8a

# Permissions
android.permissions = INTERNET,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,MANAGE_EXTERNAL_STORAGE

# Android entry point — uses the client app (connects to MAAN server)
android.entrypoint = android_main.py

# Icons (place 512x512 icon.png in repo root)
# icon.filename = %(source.dir)s/assets/icon.png
# presplash.filename = %(source.dir)s/assets/splash.png

[buildozer]
log_level = 2
warn_on_root = 1
