"""Streaming index for L5/L6 clip pickle files.

The L5 clip files are single large pickled dicts::

    {"domain": str, "videos": {video_id: video_record}}

Loading the whole file costs tens of GB of RAM for the full BDD/Dance
domains, so this module scans the pickle byte stream once (without
deserialising the payloads) and records the byte offsets of every
``videos`` value.  A single video record can then be unpickled on demand.

Only the opcodes emitted by the standard Python protocol-4/5 pickler for
this data layout (dict/list/tuple + numpy arrays) need to be understood;
values are never reconstructed during indexing.
"""
from __future__ import annotations

import os
import pickle
import pickletools
from io import BytesIO
from typing import Dict, Tuple


SCALAR_OPS = {
    "NONE", "NEWTRUE", "NEWFALSE",
    "INT", "BININT", "BININT1", "BININT2", "BININT4", "LONG1", "LONG4",
    "FLOAT", "BINFLOAT",
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8",
    "SHORT_BINSTRING", "BINSTRING",
    "BINBYTES", "SHORT_BINBYTES", "BINBYTES8",
}


def index_videos(path: str) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    """Return (domain, {video_id: (start, end)}) byte offsets.

    The segment ``[start, end)`` starts at the first opcode of a video
    record and ends right after the opcode that completes it.
    """
    with open(path, "rb") as f:
        stream = f.read()
    ops = [(op.name, arg, pos) for op, arg, pos in pickletools.genops(
        BytesIO(stream))]
    if not ops:
        return "", {}
    next_pos = [ops[i + 1][2] if i + 1 < len(ops) else len(stream)
                for i in range(len(ops))]

    frames = []        # container frames; index == creation order
    fill = []          # indices of currently open EMPTY_* frames
    stack = []         # values for REDUCE/BUILD/TUPLE bookkeeping
    domain = ""
    top_frame = -1
    videos_frame = -1
    expect_video = False
    current_videos_key = None
    expect_domain_value = False
    segments: Dict[str, Tuple[int, int]] = {}

    def cur_frame():
        return frames[fill[-1]] if fill else None

    def on_key(name: str, arg) -> None:
        nonlocal domain, expect_video, current_videos_key, expect_domain_value
        f = cur_frame()
        if f is None or f["kind"] != "dict":
            return
        if not f.get("pending_key"):
            # scalar is a key
            if isinstance(arg, bytes):
                try:
                    arg = arg.decode("utf-8")
                except UnicodeDecodeError:
                    return
            if f is frames[0]:
                if arg == "domain":
                    expect_domain_value = True
                elif arg == "videos":
                    expect_video = True
            elif f is frames[videos_frame]:
                current_videos_key = arg if isinstance(arg, str) else None
            f["pending_key"] = True
        else:
            # scalar is a value
            if f is frames[0] and expect_domain_value and \
                    isinstance(arg, (str, bytes)):
                if isinstance(arg, bytes):
                    arg = arg.decode("utf-8", "ignore")
                domain = arg
                expect_domain_value = False
            f["pending_key"] = False

    def on_value_container(fi: int) -> None:
        f = frames[fi]
        parent = fill_parent  # set by caller before pushing fi
        if parent is not None and parent["kind"] == "dict" and \
                parent.get("pending_key"):
            if videos_frame >= 0 and parent is frames[videos_frame] and \
                    f["kind"] == "dict":
                f["video_id"] = current_videos_key
            parent["pending_key"] = False

    for i, (name, arg, pos) in enumerate(ops):
        end = next_pos[i]
        if name == "STOP":
            break
        if name in ("PROTO", "FRAME"):
            continue
        if name == "MARK":
            continue
        if name in SCALAR_OPS:
            on_key(name, arg)
            stack.append(("s", None))
            continue
        if name in ("EMPTY_DICT", "EMPTY_LIST", "EMPTY_TUPLE"):
            kind = name.replace("EMPTY_", "").lower()
            fi = len(frames)
            frames.append({"kind": kind, "start": pos, "video_id": None,
                           "pending_key": False})
            fill_parent = cur_frame()
            fill.append(fi)
            stack.append(("f", fi))
            if len(frames) == 1:
                top_frame = fi
            on_value_container(fi)
            if expect_video and frames[fi]["kind"] == "dict":
                # first container value after the 'videos' key is the videos
                # dict itself; video records are its values (depth +1).
                videos_frame = fi
                expect_video = False
                # clear pending key on top-level dict
                frames[0]["pending_key"] = False
            continue
        if name == "SETITEMS":
            if fill:
                fi = fill.pop()
                f = frames[fi]
                if f.get("video_id") is not None and f["kind"] == "dict":
                    segments[f["video_id"]] = (f["start"], end)
                # completion is a value for the parent dict
                parent = cur_frame()
                if parent is not None and parent["kind"] == "dict" and \
                        parent.get("pending_key"):
                    parent["pending_key"] = False
            stack.append(("f", fi))
            continue
        if name in ("TUPLE", "LIST", "DICT", "APPENDS"):
            # aggregate opcodes complete open containers
            if name == "APPENDS" and fill:
                fi = fill.pop()
                f = frames[fi]
                if f.get("video_id") is not None and f["kind"] == "list":
                    segments[f["video_id"]] = (f["start"], end)
                stack.append(("f", fi))
            elif name in ("TUPLE", "LIST", "DICT"):
                stack.append(("s", None))
            continue
        if name in ("TUPLE1", "TUPLE2", "TUPLE3", "TUPLE4"):
            n = int(name[-1])
            for _ in range(min(n, len(stack))):
                stack.pop()
            stack.append(("s", None))
            continue
        if name == "APPEND":
            if stack:
                stack.pop()
            continue
        if name in ("BINPUT", "LONG_BINPUT", "BINPERSID", "MEMOIZE"):
            continue
        if name in ("BINGET", "LONG_BINGET", "GET"):
            stack.append(("s", None))
            continue
        if name in ("STACK_GLOBAL", "GLOBAL"):
            stack.append(("s", None))
            continue
        if name in ("REDUCE", "BUILD", "NEWOBJ", "NEWOBJ_EX"):
            n = 2 if name != "NEWOBJ_EX" else 3
            for _ in range(min(n, len(stack))):
                stack.pop()
            stack.append(("s", None))
            continue
        if name == "POP":
            if stack:
                stack.pop()
            continue
        if name == "DUP":
            if stack:
                stack.append(stack[-1])
            continue
        if name == "POP_MARK":
            continue
        # Unknown opcode: treat as scalar (best effort).
        stack.append(("s", None))

    return domain, segments


class ClipIndex:
    """Lazy per-video access to a clip pickle file."""

    def __init__(self, path: str):
        self.path = path
        self.domain, self.segments = index_videos(path)
        self._stream = None

    def _stream_bytes(self):
        if self._stream is None:
            with open(self.path, "rb") as f:
                self._stream = f.read()
        return self._stream

    def video_ids(self):
        return list(self.segments.keys())

    def load_video(self, video_id: str):
        if video_id not in self.segments:
            raise KeyError(video_id)
        start, end = self.segments[video_id]
        stream = self._stream_bytes()
        return pickle.loads(stream[start:end])

    def __len__(self):
        return len(self.segments)


def load_domain_meta(path: str):
    """Return (domain, video ids) without deserialising the payloads."""
    idx = ClipIndex(path)
    return idx.domain, idx.video_ids()


if __name__ == "__main__":
    for p in __import__("sys").argv[1:]:
        d, seg = index_videos(p)
        print(os.path.basename(p), d, len(seg), list(seg.items())[:3])
