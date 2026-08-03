"""Lucide icon helpers for AktaSense UI.

Provides inline SVG rendering of Lucide (stroke-based) icons so the app
can use real vector icons instead of emoji. Icons inherit the current
text colour via ``stroke="currentColor"``, so they adapt to buttons,
badges, headings, and cards automatically.

Design pass (V1.1) rule: no emoji in the UI — every decorative glyph is
a Lucide icon (nav/pages use Streamlit Material Symbols instead).
"""

from __future__ import annotations

import base64

# ── Lucide icon path data (24x24 viewBox, stroke-based) ─────────────────────
# Source: https://lucide.dev — each entry is the inner SVG markup.

_ICONS: dict[str, str] = {
    "home": (
        '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
        '<polyline points="9 22 9 12 15 12 15 22"/>'
    ),
    "file-text": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 '
        '2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/>'
        '<path d="M16 13H8"/><path d="M16 17H8"/>'
    ),
    "file-check": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 '
        '2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="m9 15 2 2 4-4"/>'
    ),
    "file-x": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 '
        '2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="m14.5 12.5-5 5"/>'
        '<path d="m9.5 12.5 5 5"/>'
    ),
    "file-up": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 '
        '2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M12 12v6"/>'
        '<path d="m15 15-3-3-3 3"/>'
    ),
    "folder": (
        '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9'
        'L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>'
    ),
    "upload": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>'
    ),
    "camera": (
        '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 '
        '2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>'
    ),
    "scan-text": (
        '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
        '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
        '<path d="M7 8h8"/><path d="M7 12h10"/><path d="M7 16h6"/>'
    ),
    "sparkles": (
        '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 '
        '0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912'
        'a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M19 17v4"/>'
        '<path d="M3 5h4"/><path d="M17 19h4"/>'
    ),
    "check-circle": (
        '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>'
    ),
    "x-circle": (
        '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>'
    ),
    "alert-triangle": (
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 '
        '0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>'
    ),
    "info": (
        '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>'
    ),
    "download": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>'
    ),
    "refresh-cw": (
        '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>'
        '<path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>'
        '<path d="M3 21v-5h5"/>'
    ),
    "rotate-ccw": (
        '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>'
        '<path d="M3 3v5h5"/>'
    ),
    "target": (
        '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/>'
        '<circle cx="12" cy="12" r="2"/>'
    ),
    "bar-chart-3": (
        '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M18 17V9"/>'
        '<path d="M13 17V5"/><path d="M8 17v-3"/>'
    ),
    "search": (
        '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>'
    ),
    "lightbulb": (
        '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 '
        '.2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/>'
        '<path d="M10 22h4"/>'
    ),
    "list": (
        '<path d="M3 12h.01"/><path d="M3 18h.01"/><path d="M3 6h.01"/>'
        '<path d="M8 12h13"/><path d="M8 18h13"/><path d="M8 6h13"/>'
    ),
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "database": (
        '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/>'
        '<path d="M3 12A9 3 0 0 0 21 12"/>'
    ),
    "arrow-right": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "cpu": (
        '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
        '<path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/>'
        '<path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>'
    ),
    "layers": (
        '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 '
        '0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/>'
        '<path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>'
    ),
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "clock": (
        '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'
    ),
    "shield-check": (
        '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 '
        '4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>'
        '<path d="m9 12 2 2 4-4"/>'
    ),
    "badge-check": (
        '<path d="M3.85 8.62a4 4 0 0 1 4.78-4.77 4 4 0 0 1 6.74 0 4 4 0 0 1 4.78 4.78 4 4 0 0 1 0 '
        '6.74 4 4 0 0 1-4.77 4.78 4 4 0 0 1-6.75 0 4 4 0 0 1-4.78-4.77 4 4 0 0 1 0-6.76Z"/>'
        '<path d="m9 12 2 2 4-4"/>'
    ),
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "file-stack": (
        '<path d="M21 7h-3a2 2 0 0 1-2-2V2"/><path d="M21 6v6.5c0 .8-.7 1.5-1.5 1.5h-7c-.8 0-1.5-.7-1.5-1.5v-9c0-.8.7-1.5 1.5-1.5H17Z"/>'
        '<path d="M7 8v8.8c0 .3.2.6.4.8.2.2.5.4.8.4H15"/><path d="M3 12v8.8c0 .3.2.6.4.8.2.2.5.4.8.4H11"/>'
    ),
    "globe": (
        '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>'
        '<path d="M2 12h20"/>'
    ),
}


def icon(name: str, size: int = 16, class_: str = "") -> str:
    """Return an inline Lucide SVG for ``name``.

    Args:
        name: Lucide icon key (see ``_ICONS``).
        size: Pixel size (width/height).
        class_: Optional extra CSS class for the SVG element.

    Returns:
        SVG markup string, or an empty string if the icon is unknown.
    """
    paths = _ICONS.get(name)
    if paths is None:
        return ""
    cls = f' class="{class_}"' if class_ else ""
    return (
        f'<svg{cls} width="{size}" height="{size}" viewBox="0 0 24 24" '
        'xmlns="http://www.w3.org/2000/svg" role="img" '
        'fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true" preserveAspectRatio="xMidYMid meet">'
        f"{paths}</svg>"
    )


def icon_markdown(name: str, size: int = 16, alt: str = "icon") -> str:
    """Return a markdown image tag for the Lucide icon so Streamlit buttons and headers render it."""
    paths = _ICONS.get(name)
    if paths is None:
        return ""
    svg = (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" '
        'xmlns="http://www.w3.org/2000/svg" role="img" '
        'fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true" preserveAspectRatio="xMidYMid meet">'
        f"{paths}</svg>"
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'![{alt}](data:image/svg+xml;base64,{encoded})'


def inline_icon(name: str, size: int = 14) -> str:
    """Return an icon wrapped in the ``.akta-icon-inline`` span (vertical align)."""
    return f'<span class="akta-icon-inline">{icon(name, size)}</span>'


def icon_heading(name: str, text: str, size: int = 20) -> str:
    """Return a styled page-title block: Lucide icon + bold heading text."""
    return (
        f'<div class="akta-page-title">{icon(name, size)}<span>{text}</span></div>'
    )
