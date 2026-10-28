"""Server entrypoint for the Research Intelligence Engine FastAPI backend."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
