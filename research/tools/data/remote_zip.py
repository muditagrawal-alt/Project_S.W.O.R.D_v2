"""
Read members of a large remote zip over HTTP range requests (no full download), with TLS verification.

    from remote_zip import open_remote_zip
    z = open_remote_zip(url, ca_bundle)      # zipfile.ZipFile
    z.infolist(); z.extract(member, path)

Used for UCF-Crime (one 103 GB archive): only the needed videos and annotation files are fetched.
"""

from __future__ import annotations

import io
import zipfile

import requests


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str, session: requests.Session):
        self.url, self.sess, self.pos = url, session, 0
        r = session.head(url, timeout=60, allow_redirects=True)
        r.raise_for_status()
        if r.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise IOError("server does not support range requests")
        self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = off if whence == io.SEEK_SET else self.pos + off if whence == io.SEEK_CUR else self.size + off
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        r = self.sess.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}, timeout=300)
        r.raise_for_status()
        data = r.content
        if r.status_code != 206 or len(data) != n:
            raise IOError(f"range request returned {r.status_code}, {len(data)} of {n} bytes")
        b[:n] = data
        self.pos += n
        return n


def open_remote_zip(url: str, ca_bundle: str | bool = True, buffer_size: int = 8 << 20) -> zipfile.ZipFile:
    sess = requests.Session()
    sess.verify = ca_bundle
    return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url, sess), buffer_size=buffer_size))
