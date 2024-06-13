import os

import juliapkg

_FINCH_NAME = "Finch"
_FINCH_VERSION = "0.6.31"
_FINCH_HASH = "9177782c-1635-4eb9-9bfb-d9dfa25e6bce"
_FINCH_REPO_PATH = os.environ.get("FINCH_REPO_PATH", default=None)
_FINCH_REPO_URL = os.environ.get("FINCH_URL_PATH", default=None)

_TENSOR_MARKET_NAME = "TensorMarket"
_TENSOR_MARKET_HASH = "8b7d4fe7-0b45-4d0d-9dd8-5cc9b23b4b77"
_TENSOR_MARKET_VERSION = "0.2.0"

if _FINCH_REPO_PATH and _FINCH_REPO_URL:
    raise ValueError("FINCH_REPO_PATH and FINCH_URL_PATH can't be set at the same time.")

deps = juliapkg.deps.load_cur_deps()

if _FINCH_REPO_PATH:  # Also account for empty string
    juliapkg.add(_FINCH_NAME, _FINCH_HASH, path=_FINCH_REPO_PATH, dev=True)
elif _FINCH_REPO_URL:
    juliapkg.add(_FINCH_NAME, _FINCH_HASH, url=_FINCH_REPO_URL, dev=True)
elif (
    deps.get("packages", {}).get(_FINCH_NAME, {}).get("version", None)
    != _FINCH_VERSION
):
    juliapkg.add(_FINCH_NAME, _FINCH_HASH, version=_FINCH_VERSION)

if (
    deps.get("packages", {}).get(_TENSOR_MARKET_NAME, {}).get("version", None)
    != _FINCH_VERSION
):
    juliapkg.add(_TENSOR_MARKET_NAME, _TENSOR_MARKET_HASH, version=_TENSOR_MARKET_VERSION)

import juliacall as jc  # noqa

juliapkg.resolve()
from juliacall import Main as jl  # noqa

jl.seval("using Finch")
jl.seval("using Random")
jl.seval("using TensorMarket")
