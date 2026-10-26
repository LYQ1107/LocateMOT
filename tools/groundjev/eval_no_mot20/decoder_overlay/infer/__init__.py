"""Use the pinned official infer package with one request-seed scope repair."""
import os
from pathlib import Path

__path__.append(str(Path(os.environ['GROUNDJEV_OFFICIAL_SOURCE'])/'infer'))
