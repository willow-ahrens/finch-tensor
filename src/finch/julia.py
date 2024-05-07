import juliapkg

_FINCH_VERSION = "0.6.24"
_FINCH_HASH = "9177782c-1635-4eb9-9bfb-d9dfa25e6bce"

deps = juliapkg.deps.load_cur_deps()
if deps.get("packages", {}).get("Finch", {}).get("version", None) != _FINCH_VERSION:
    juliapkg.add("Finch", _FINCH_HASH, version=_FINCH_VERSION)

import juliacall  # noqa

juliapkg.resolve()
from juliacall import Main as jl  # noqa

jl.seval("using Finch")
jl.seval("using Random")
