"""python -m service [--host 127.0.0.1] [--port 8000]  — start the S.W.O.R.D. service (dashboard at /)."""

import argparse
import importlib.util
import sys

NEEDED = ("uvicorn", "fastapi", "sqlalchemy", "torch", "cv2", "ultralytics", "rtmlib", "lap", "supervision")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 unless authentication is added")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    missing = [m for m in NEEDED if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit(f"{sys.executable} is missing {', '.join(missing)}: run the service from the project's virtual "
                 "environment (source venv/bin/activate) or pip install -r requirements.txt")
    import uvicorn
    uvicorn.run("service.api:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
