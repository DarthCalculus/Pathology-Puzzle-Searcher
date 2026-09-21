"""
Pytest config: put the repo dir on sys.path so tests can import the
harvest modules.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
