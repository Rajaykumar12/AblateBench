import sys
from pathlib import Path

# Make src/ importable as top-level modules (extract_direction, ablate_runtime, ...)
# without needing an installed package — matches how modal_app.py will import them
# too (mounted alongside the entrypoint on the Modal container).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
