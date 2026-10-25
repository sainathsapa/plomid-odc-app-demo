"""PLOMID startup banner (copper ASCII logo + version).

Printed to stderr on `odc serve` so piped stdout stays machine-clean.
Colors only on real terminals (honors NO_COLOR); anywhere else plain text.
"""

from __future__ import annotations

import os
import sys

# Brand copper. Matches --accent (#e2683c) in odc/console/static/style.css.
COPPER_RGB = (226, 104, 60)

LOGO_LINES = [
    "    **      **********     **             ********      **        ***   **    **********",
    " ********            **    **           ***      ***    ****    *****   **             ***",
    "***    ***           **    **          **         **    ** ******* **   **    **        **",
    "***  *****  **********     **          **         **    **   **    **   **    **        **",
    "***  **     **             **           ***      ***    **         **   **    **       ***",
    "  *         **             **********     ********      **         **   **    **********  ",
]


def supports_color(stream: object | None = None) -> bool:
    """True only for live color terminals (NO_COLOR kills it). Pure."""
    if os.environ.get("NO_COLOR"):
        return False
    target = stream if stream is not None else sys.stderr
    return bool(getattr(target, "isatty", lambda: False)())


def render_banner(color: bool | None = None) -> str:
    """Full startup screen text. Pure — pass color explicitly in tests."""
    use_color = supports_color() if color is None else color
    lines = list(LOGO_LINES)
    if use_color:
        r, g, b = COPPER_RGB
        start, reset = f"\033[38;2;{r};{g};{b}m", "\033[0m"
        lines = [f"{start}{line}{reset}" for line in lines]
    from . import __version__

    lines.append(f"PLOMID ODC — occupant console v{__version__}")
    return "\n".join(lines) + "\n"


def print_banner() -> None:
    """Write the startup screen to stderr (never stdout)."""
    sys.stderr.write(render_banner())
    sys.stderr.flush()
