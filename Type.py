"""
This module provides standard scheme for solving the discrete/lattice Boltzmann equation

"""
from typing import TypeAlias
from typing import TypedDict
import jax

Array: TypeAlias = jax.Array
Lattice = TypedDict(
    "Lattice",
    {
        "Name": str,
        "XI": Array,
        "XIT": Array,
        "C": Array,
        "W": Array,
        "OPP": Array,
        "CS": float,
    },
)


DVS = TypedDict(
    "DVS",
    {"Name": str, "XI": Array, "XIT": Array, "W": Array, "OPP": Array},
)

