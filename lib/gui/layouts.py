"""
GUI Layout Helpers
------------------
Utilities for creating consistent, optimized GUI layouts with proper spacing and grouping.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable

from .styles import STANDARD_PADDING


def create_section(parent, title: str, padding: str = "medium") -> ttk.LabelFrame:
    """
    Create a labeled section frame with consistent styling.
    
    Args:
        parent: Parent widget
        title: Section title
        padding: Padding size ("small", "medium", "large", "xlarge")
    
    Returns:
        LabelFrame widget
    """
    pad = STANDARD_PADDING.get(padding, STANDARD_PADDING["medium"])
    return ttk.LabelFrame(parent, text=title, padding=pad)


def create_action_buttons(parent, buttons: list[tuple[str, Callable, Optional[str]]], 
                          fill: str = "x", spacing: int = 2) -> list[ttk.Button]:
    """
    Create a row of action buttons with consistent styling.
    
    Args:
        parent: Parent widget
        buttons: List of (text, command, tooltip) tuples
        fill: Fill option ("x", "y", "both", "none")
        spacing: Padding between buttons
    
    Returns:
        List of created Button widgets
    """
    frame = ttk.Frame(parent)
    frame.pack(fill=fill, pady=(0, spacing))
    
    created_buttons = []
    for text, command, tooltip_text in buttons:
        btn = ttk.Button(frame, text=text, command=command)
        btn.pack(side="left", fill="x", expand=True, padx=(0, spacing))
        if tooltip_text:
            from .common import tooltip
            tooltip(btn, tooltip_text)
        created_buttons.append(btn)
    
    return created_buttons


def create_labeled_entry(parent, label: str, default_value: str = "", 
                        width: int = 15, tooltip: Optional[str] = None) -> tuple[ttk.Label, ttk.Entry]:
    """
    Create a labeled entry field with consistent layout.
    
    Args:
        parent: Parent widget
        label: Label text
        default_value: Default entry value
        width: Entry width
        tooltip: Optional tooltip text
    
    Returns:
        Tuple of (Label, Entry) widgets
    """
    frame = ttk.Frame(parent)
    frame.pack(fill="x", pady=2)
    
    lbl = ttk.Label(frame, text=label, width=18, anchor="w")
    lbl.pack(side="left", padx=(0, 8))
    
    entry = ttk.Entry(frame, width=width)
    entry.pack(side="left")
    if default_value:
        entry.insert(0, default_value)
    
    if tooltip:
        from .common import tooltip as add_tooltip
        add_tooltip(entry, tooltip)
    
    return lbl, entry


def create_button_group(parent, title: Optional[str] = None, 
                       orientation: str = "horizontal") -> ttk.Frame:
    """
    Create a frame for grouping related buttons.
    
    Args:
        parent: Parent widget
        title: Optional section title
        orientation: "horizontal" or "vertical"
    
    Returns:
        Frame widget (or LabelFrame if title provided)
    """
    if title:
        frame = ttk.LabelFrame(parent, text=title, padding=STANDARD_PADDING["small"])
    else:
        frame = ttk.Frame(parent)
    
    return frame

