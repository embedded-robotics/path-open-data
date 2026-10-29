#!/usr/bin/env bash
# Compile every generated table in isolation to catch overfull boxes and missing packages.
# Structural checks cannot see page overflow - Table 1 ran 243pt off the page and looked
# fine until it was typeset.
#
# NOTE: this harness uses plain `article`, not the Springer `sn-jnl` class, so it cannot
# reproduce the class-specific trap that sn-jnl's `table' environment silently DROPS a
# float whose body is a box (losing the caption and \label). After changing a wide table,
# also build the real manuscript and grep the PDF for '??'.
#
# Needs `tectonic` on PATH (single binary, fetches only the packages a document uses;
# a full TeX Live will not fit - /data is at 100%).
#   curl -L -o tectonic.tar.gz \
#     https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz
#   tar xzf tectonic.tar.gz
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${1:-$(mktemp -d)}"
mkdir -p "$BUILD"
cp "$HERE"/tables/*.tex "$BUILD"/
{
  echo '\documentclass[11pt]{article}'
  echo '\usepackage[margin=18mm]{geometry}'
  echo '\usepackage{booktabs}\usepackage{siunitx}\usepackage{graphicx}'
  # Wide tables are built into a savebox before their float, because a scaling box cannot
  # sit inside the Springer class's `table' environment (see the manuscript BUILD_NOTES).
  echo '\newsavebox{\pathopentablebox}'
  # The real manuscript uses the class's unwrapped `tableorg' float for those tables;
  # plain article has no such environment, so alias it to `table' for this check.
  echo '\newenvironment{tableorg}[1][]{\begin{table}[#1]}{\end{table}}'
  # \TODO marks values pending data deposition; harmless here, but must be defined.
  echo '\newcommand{\TODO}[1]{[TODO: #1]}'
  echo '\begin{document}'
  for f in "$HERE"/tables/*.tex; do
    echo "\\input{$(basename "$f")}"; echo '\clearpage'
  done
  echo '\end{document}'
} > "$BUILD/build.tex"
cd "$BUILD"
tectonic -k build.tex 2>&1 | grep -iE 'overfull|underfull|error|warning' | sort -u || true
echo "--- built: $BUILD/build.pdf"
