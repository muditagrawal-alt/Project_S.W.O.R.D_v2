"""python -m service [--host 127.0.0.1] [--port 8000]  — start the S.W.O.R.D. service (dashboard at /)."""

import argparse

import uvicorn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 unless authentication is added")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run("service.api:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
