[app]
title = Race Planner Mobile
package.name = raceplanmobile
package.domain = org.trailanalyzer

source.dir = .
source.include_exts = py,png,jpg,kv,atlas
source.include_patterns = RacePlanMobile_Kivy.py

version = 1.0

requirements = python3,kivy==2.3.0,numpy,pandas,scipy,gpxpy

orientation = portrait
fullscreen = 0

android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_IMAGES,READ_MEDIA_VIDEO,READ_MEDIA_AUDIO
android.api = 33
android.minapi = 26
android.ndk = 25b
android.archs = arm64-v8a, armeabi-v7a

android.allow_backup = True

[buildozer]
log_level = 2
warn_on_root = 1
