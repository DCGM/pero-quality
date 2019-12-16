#!/usr/bin/env python3

from io import BytesIO
from os.path import dirname
from urllib3 import PoolManager, HTTPResponse
from zipfile import ZipFile


def main():
    http = PoolManager()

    r: HTTPResponse = http.request('GET', 'http://www.fit.vutbr.cz/~ihradis/pero-models/ocr_quality_LN_2019-12-16.zip')

    assert 200 <= r.status < 300, f"Request status {r.status}"

    zfile = ZipFile(BytesIO(r.data))
    zfile.extractall(path=dirname(__file__))


if __name__ == "__main__":
    main()
