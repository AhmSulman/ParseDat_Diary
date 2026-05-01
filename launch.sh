#!/usr/bin/env bash
set -e
echo ""
echo " MAAN — Chat with Books"
echo " RTX 4050 · Local AI · Your Data"
echo ""
[ ! -f .deps_ok ] && pip3 install -r requirements.txt && touch .deps_ok
python3 main.py "$@"
