#!/usr/bin/env bash
set -e
binary=/usr/bin/grep.neuron-paper-original
[[ -x "$binary" ]] || binary=/usr/bin/grep
exec /usr/bin/timeout --signal=TERM --kill-after=2s 60s "$binary" --binary-files=without-match "$@"
