"""New immutable-run entry point. Never uploads anything."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
if __name__=='__main__':
    from goid_v4.official_aligned_v1 import main
    main()
