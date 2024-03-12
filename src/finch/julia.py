import juliapkg

juliapkg.add("Finch", "9177782c-1635-4eb9-9bfb-d9dfa25e6bce", version="0.6.16")
import juliacall  # noqa

juliapkg.resolve()
from juliacall import Main as jl  # noqa

jl.seval("using Finch")
