#!/usr/bin/env python3
"""Smoke test for cache fixes - 2 videos only."""
import sys
sys.path.insert(0, "/root/scripts")

# Patch MAX_VIDEOS to 2
import test_cache_eval_10videos
test_cache_eval_10videos.MAX_VIDEOS = 2

if __name__ == "__main__":
    sys.exit(test_cache_eval_10videos.main())
