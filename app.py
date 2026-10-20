"""Root entrypoint for Streamlit Web Application.

Allows running either:
    streamlit run app.py
or:
    streamlit run src/ui/app.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import and execute the UI application
from src.ui.app import *

