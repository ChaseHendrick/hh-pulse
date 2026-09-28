#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
"""Load code/hypotheses.json and scan paper/paper.md for priority phrases.

An unread item blocks an unqualified priority phrase. The sentence "we found no
earlier proof, within them" is the paper's own limit, so that one phrase does
not fail when "within them" is still in the paper. A numerical remark is not
an unread hypothesis. Exit 1 if some other priority phrase occurs while any
item is unread, or if "within them" has been removed.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
HYPO = os.path.join(ROOT, "code", "hypotheses.json")
PAPER = os.path.join(ROOT, "paper", "paper.md")
PHRASES = ("no earlier", "first proof", "has not been proved", "for the first time")


def main():
    hypotheses = json.load(open(HYPO))
    missing = []
    for item in hypotheses:
        raw = item.get("certificate") or ""
        for rel in [part.strip() for part in raw.split(",") if part.strip()]:
            if not os.path.isfile(os.path.join(ROOT, rel)):
                missing.append("%s -> %s" % (item.get("id"), rel))
    if missing:
        print("certificate file missing")
        for line in missing:
            print(line)
        return 1
    text = re.sub(r"\s+", " ", open(PAPER).read())
    found = [p for p in PHRASES if p in text]
    unread = [h["id"] for h in hypotheses if h.get("status") == "unread"]
    counts = {"boxed": 0, "cited": 0, "unread": 0, "numerical": 0}
    for h in hypotheses:
        counts[h["status"]] = counts.get(h["status"], 0) + 1
    qualified = "within them" in text
    blocking = []
    for phrase in found:
        if phrase == "no earlier" and qualified:
            continue
        blocking.append(phrase)
    if "no earlier" in found and not qualified:
        print("the earlier-work sentence no longer says within them")
        print("unread: " + ", ".join(unread))
        return 1
    if unread and blocking:
        for phrase in blocking:
            print(phrase)
        print("unread: " + ", ".join(unread))
        return 1
    if not found:
        print("no such phrase occurs")
    else:
        print("priority phrase found: " + "; ".join(found))
        if qualified:
            print("the earlier-work sentence keeps the limit: within them")
    print("boxed %d / cited %d / unread %d / numerical %d" % (
        counts["boxed"], counts["cited"], counts["unread"], counts["numerical"]))
    if unread:
        print("unread: " + ", ".join(unread))
    return 0


if __name__ == "__main__":
    sys.exit(main())
