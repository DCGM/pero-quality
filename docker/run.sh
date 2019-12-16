#!/usr/bin/env bash

# --rm -> remove container after exitting
# -u $(id -u):$(id -g) -> map user id and group id (workaround for not running container as root)
# -t -> open pseudo-tty and connect it to stdin
# -i -> keep STDIN open even if not attached
# -v host/dir:container/dir -> map host directory in cointainer
docker run --rm -i -t -v $(realpath ..):/pero -u $(id -u):$(id -g) pero-ocr  /bin/bash
