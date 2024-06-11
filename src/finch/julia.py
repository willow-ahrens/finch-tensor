import os

import juliapkg

_FINCH_NAME = "Finch"
_FINCH_VERSION = "0.6.31"
_FINCH_HASH = "9177782c-1635-4eb9-9bfb-d9dfa25e6bce"
_FINCH_REPO_PATH = os.environ.get("FINCH_REPO_PATH", default=None)

if _FINCH_REPO_PATH:  # Also account for empty string
    juliapkg.add(_FINCH_NAME, _FINCH_HASH, path=_FINCH_REPO_PATH, dev=True)
else:
    deps = juliapkg.deps.load_cur_deps()
    if (
        deps.get("packages", {}).get(_FINCH_NAME, {}).get("version", None)
        != _FINCH_VERSION
    ):
        juliapkg.add(_FINCH_NAME, _FINCH_HASH, version=_FINCH_VERSION)

import juliacall as jc  # noqa

juliapkg.resolve()
from juliacall import Main as jl  # noqa

jl.seval("using Finch")
jl.seval("using Random")
