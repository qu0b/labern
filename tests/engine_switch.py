#!/usr/bin/env python3
"""Unit test for the tray engine switcher.

Checks the thing that makes runtime switching safe: the binding dicts the tray
mutates are the *same objects* the keymap resolves at dictation time, so a
switch lands on the next dictation without a restart.

No audio device, no tray, no network — VoiceInput is built with tray off and the
context listener disabled.

Run:  .venv/bin/python tests/engine_switch.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import voice_input as v  # noqa: E402

FAILED = []


def check(cond, what):
    print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        FAILED.append(what)


def build():
    return v.VoiceInput(
        bindings=[dict(b) for b in v.BINDINGS],
        model_size="tiny", device="cpu", use_tray=False, initial_prompt=None,
        remote_url="http://example.invalid/v1/audio/transcriptions", api_key="k",
        run_context_listener=False,
        engine_catalog=v.ENGINE_CATALOG,
    )


print("_apply_engine")
b = {"label": "raw", "model": "whisper-large-v3-turbo", "language": None}
v._apply_engine(b, {"model": "parakeet-tdt-0.6b-v2", "language": "en"})
check(b["model"] == "parakeet-tdt-0.6b-v2", "model replaced")
check(b["language"] == "en", "language replaced")

# A terse entry must not wipe fields it doesn't mention.
b2 = {"label": "raw", "model": "a", "url": "http://keep.me", "provider": "openai"}
v._apply_engine(b2, {"model": "b"})
check(b2["url"] == "http://keep.me", "unmentioned url preserved")
check(b2["provider"] == "openai", "unmentioned provider preserved")

# Empty-string language means auto-detect, i.e. None on the binding.
b3 = {"label": "raw", "model": "a", "language": "en"}
v._apply_engine(b3, {"model": "a", "language": ""})
check(b3["language"] is None, "empty language normalises to None (auto-detect)")

# api_key_env resolves from the environment.
os.environ["_LABERN_TEST_KEY"] = "sk-test"
b4 = {"label": "raw", "model": "a"}
v._apply_engine(b4, {"model": "ink-2", "api_key_env": "_LABERN_TEST_KEY"})
check(b4.get("api_key") == "sk-test", "api_key_env resolved from environment")

print("startup override path uses the same helper")
vi = build()
check(all("keyobj" in x for x in vi.bindings), "bindings resolved")

print("switch mutates the object the keymap holds")
vi = build()
target = vi.bindings[0]
keymap_obj = vi._keymap[target["keyobj"]]
check(keymap_obj is target, "keymap and bindings share one dict")

before = target["model"]
vi._switch_engine(target, {"model": "parakeet-tdt-0.6b-v2", "language": "en"}, "parakeet")
check(target["model"] == "parakeet-tdt-0.6b-v2", f"binding switched from {before}")
check(vi._keymap[target["keyobj"]]["model"] == "parakeet-tdt-0.6b-v2",
      "keymap sees the switch (no restart needed)")
check(vi.bindings[1]["model"] == before, "other keys untouched")

print("use_remote recomputed on switch")
vi = build()
vi._switch_engine(vi.bindings[0], {"model": "x"}, "x")
check(vi.use_remote is True, "still remote while a remote url+key remain")

print("catalog includes whatever a key starts on")
# Mirrors the main() logic: a configured engine absent from the catalog is added
# so the radio menu can show the live selection as checked.
catalog = dict(v.ENGINE_CATALOG)
known = {e.get("model") for e in catalog.values()}
custom = {"label": "raw", "model": "ink-2", "provider": "cartesia"}
if custom["model"] not in known:
    catalog[f"{custom['model']} (current)"] = {
        k: custom[k] for k in v.ENGINE_FIELDS if k in custom
    }
check("ink-2 (current)" in catalog, "unlisted current engine added to catalog")
check(catalog["ink-2 (current)"]["provider"] == "cartesia", "its provider carried over")

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}):")
    for f in FAILED:
        print("  -", f)
    sys.exit(1)
print("all engine-switch checks passed")
