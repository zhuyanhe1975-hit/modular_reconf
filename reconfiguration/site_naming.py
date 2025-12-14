#!/usr/bin/env python3
"""
Site naming utilities for UBot model.
"""

from .connection_graph import SiteRef


def site_full_name(site_ref: SiteRef) -> str:
    """Convert SiteRef to XML site name, e.g. ma_right -> ma_connector_right."""
    return f"{site_ref.half}_connector_{site_ref.site}"


def parse_site_ref(site_full_name: str) -> SiteRef:
    """Parse 'ma_connector_right' to SiteRef."""
    parts = site_full_name.replace('_connector_', '_').split('_', 2)
    half, site = parts[1], parts[2] if len(parts) > 2 else parts[1]
    return SiteRef(module_id=-1, half=half, site=site.replace('connector_', '') if 'connector' in site else site)


# Quick lookup dict
SITE_NAMES = {
    ("ma", "right"): "ma_connector_right",
    ("ma", "bottom"): "ma_connector_bottom",
    ("mb", "left"): "mb_connector_left",
    ("mb", "top"): "mb_connector_top"
}


def get_site_full_name(half: str, site: str) -> str:
    """Get full XML name from half and site str."""
    return SITE_NAMES.get((half, site), f"{half}_connector_{site}")


# Hardcode for Phase-2.4
def xml_site_to_pair(xml_site_name: str):
    """Return (half, site_type) for Phase-2.4 hardcoded."""
    mapping = {
        "ma_connector_right": ("ma", "right"),
        "ma_connector_bottom": ("ma", "bottom"),
        "mb_connector_left": ("mb", "left"),
        "mb_connector_top": ("mb", "top")
    }
    return mapping.get(xml_site_name, ("unknown", "unknown"))
