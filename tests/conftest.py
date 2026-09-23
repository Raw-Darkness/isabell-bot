"""Run against the example config in a throwaway directory: no secrets, no real data."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["BOT_CONFIG"] = os.path.join(ROOT, "Config.example.json")
sys.path.insert(0, ROOT)
os.chdir(tempfile.mkdtemp(prefix="isabell-test-"))
