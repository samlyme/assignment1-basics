import pprint
import textwrap
from collections.abc import Mapping


def format_columns(
    value: object,
    right_text: str = "",
    *,
    left_width: int = 80,
    right_width: int = 25,
    gap: int = 3,
) -> str:
    """Format an object in a wrapped left column with text on the right."""

    # Convert defaultdict and other mapping subclasses for cleaner output.
    if isinstance(value, Mapping):
        value = dict(value)

    formatted = pprint.pformat(
        value,
        width=left_width,
        compact=False,
        sort_dicts=False,
    )

    left_lines: list[str] = []
    for line in formatted.splitlines():
        # Also wrap unusually long strings or values.
        wrapped = textwrap.wrap(
            line,
            width=left_width,
            subsequent_indent="  ",
            replace_whitespace=False,
            drop_whitespace=False,
        )
        left_lines.extend(wrapped or [""])

    right_lines = textwrap.wrap(right_text, width=right_width) or [""]

    row_count = max(len(left_lines), len(right_lines))
    left_lines.extend([""] * (row_count - len(left_lines)))
    right_lines.extend([""] * (row_count - len(right_lines)))

    separator = " " * gap

    return "\n".join(f"{left:<{left_width}}{separator}{right}" for left, right in zip(left_lines, right_lines))
