#!/usr/bin/env python3

import sys
import os

def main():
    # Add src directory to path
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Import and run comprehensive evaluation
    from eval_visualize import run_all_evaluations
    run_all_evaluations()

if __name__ == '__main__':
    main()
