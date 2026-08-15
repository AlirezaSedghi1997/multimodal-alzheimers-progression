import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# An outer checkout may already be injected into sys.path by the desktop test
# runtime.  Promote this tree deterministically without deleting modules.
while str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
