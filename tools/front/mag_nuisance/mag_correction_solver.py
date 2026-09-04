"""Compatibility imports for the production magnetic-nuisance core.

New code should import :mod:`backend.mag_nuisance_core` directly. This module
is retained so older analysis scripts and notebooks continue to work.
"""

from backend.mag_nuisance_core import *  # noqa: F401,F403
