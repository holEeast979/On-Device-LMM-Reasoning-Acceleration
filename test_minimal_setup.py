#!/usr/bin/env python3
"""Minimal test: can load_model() return normally?"""
import sys
sys.stdout.reconfigure(line_buffering=True)

print("step1: importing...", flush=True)
from fasteromni.pipeline import SparseInferencePipeline

print("step2: creating pipeline...", flush=True)
pipe = SparseInferencePipeline()

print("step3: loading model...", flush=True)
pipe.load_model()

print("step4: model loaded OK", flush=True)
print("step5: _model=%s, _proc=%s" % (type(pipe._model).__name__, type(pipe._proc).__name__), flush=True)

print("step6: importing encoder_cache...", flush=True)
from fasteromni.encoder_cache import EncoderCacheHook, patch_pipeline_run_inference

print("step7: creating cache hook...", flush=True)
cache = EncoderCacheHook(pipe._model, pipe._proc)

print("step8: enabling hook...", flush=True)
cache.enable()

print("step9: hook enabled=%s" % cache.hook_enabled, flush=True)

print("step10: importing EncoderTimer...", flush=True)
from utils.profiling_utils import EncoderTimer

print("step11: creating timer...", flush=True)
timer = EncoderTimer()
timer.register(pipe._model)

print("step12: all setup done, ready to run inference", flush=True)
print("ALL STEPS PASSED", flush=True)
