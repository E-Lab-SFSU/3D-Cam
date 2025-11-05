"""
Common GUI Widgets and Utilities
---------------------------------
Shared widgets and utility functions for GUI components.
"""

import tkinter as tk
from tkinter import ttk
from typing import Union


def tooltip(widget: Union[tk.Widget, tk.Toplevel], text: str) -> None:
    """
    Create a tooltip for a widget that appears on hover.
    
    Args:
        widget: The tkinter widget to attach the tooltip to
        text: The tooltip text to display
    
    Example:
        >>> button = ttk.Button(root, text="Click me")
        >>> tooltip(button, "This button does something")
    """
    tip = tk.Toplevel(widget)
    tip.withdraw()
    tip.overrideredirect(True)
    lbl = tk.Label(
        tip,
        text=text,
        bg="#ffffe0",
        relief="solid",
        borderwidth=1,
        font=("TkDefaultFont", 9)
    )
    lbl.pack()
    
    def show(_):
        """Show the tooltip when mouse enters the widget."""
        tip.geometry(f"+{widget.winfo_rootx()+30}+{widget.winfo_rooty()+10}")
        tip.deiconify()
    
    def hide(_):
        """Hide the tooltip when mouse leaves the widget."""
        tip.withdraw()
    
    widget.bind("<Enter>", show)
    widget.bind("<Leave>", hide)


class ScrollableFrame(ttk.Frame):
    """
    A scrollable frame widget that can contain any number of widgets.
    
    Usage:
        >>> root = tk.Tk()
        >>> scrollable = ScrollableFrame(root)
        >>> scrollable.pack(fill="both", expand=True)
        >>> # Add widgets to scrollable.inner_frame
        >>> button = ttk.Button(scrollable.inner_frame, text="Click me")
        >>> button.pack()
    """
    
    def __init__(self, parent, *args, **kwargs):
        """Initialize the scrollable frame."""
        # Create outer frame
        super().__init__(parent, *args, **kwargs)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        # Create inner frame that will be scrollable
        self.inner_frame = ttk.Frame(self.canvas)
        
        # Configure canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Update scroll region when inner frame changes size
        def configure_scroll_region(event=None):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        self.inner_frame.bind("<Configure>", configure_scroll_region)
        
        # Update canvas width when outer frame changes size
        def configure_canvas_width(event=None):
            canvas_width = event.width
            self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        
        self.canvas.bind("<Configure>", configure_canvas_width)
        
        # Bind mouse wheel for scrolling
        def on_mousewheel(event):
            # Windows and MacOS
            if event.delta:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            # Linux
            elif event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")
        
        def bind_mousewheel(event):
            self.canvas.bind_all("<MouseWheel>", on_mousewheel)
            self.canvas.bind_all("<Button-4>", on_mousewheel)
            self.canvas.bind_all("<Button-5>", on_mousewheel)
        
        def unbind_mousewheel(event):
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
        
        self.canvas.bind("<Enter>", bind_mousewheel)
        self.canvas.bind("<Leave>", unbind_mousewheel)
        self.inner_frame.bind("<Enter>", bind_mousewheel)
        self.inner_frame.bind("<Leave>", unbind_mousewheel)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

