#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path[:0]=['/control',str(Path(__file__).resolve().parents[3]/'tools')]
from rpc_client import main,call
if __name__=='__main__':main('run_tests')
