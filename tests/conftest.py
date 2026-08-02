import os


# Production defaults the live market loop to on.  Tests explicitly disable
# background network threads before test modules import the Flask application.
os.environ.setdefault("START_LIVE_LOOP_ON_BOOT", "0")
