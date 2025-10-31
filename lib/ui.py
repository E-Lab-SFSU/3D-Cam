import tkinter as tk
from tkinter import ttk


def set_controls_enabled(widgets: dict, enabled: bool):
    state = "normal" if enabled else "disabled"
    for w in widgets.values():
        try:
            w.configure(state=state)
        except Exception:
            pass


def build_gui(
    params: dict,
    overlays: dict,
    overlay_targets: dict,
    on_open_video,
    on_export_video,
    on_optimize_center,
    on_reset,
    on_exit,
    on_toggle_play_pause=None,  # Optional, not used in main window anymore
):
    widgets: dict = {}
    gui_vars_numeric: dict = {}
    gui_vars_check: dict = {}

    root = tk.Tk()
    root.title("Pair Detector v4.5 — Controls")
    root.geometry("540x950+60+60")
    root.resizable(True, True)

    s = ttk.Style(root)
    try:
        s.theme_use("clam")
    except Exception:
        pass

    # Main content frame (no scrolling - everything fits)
    content_frame = ttk.Frame(root)
    content_frame.pack(fill="both", expand=True, padx=8, pady=6)

    # Top row buttons (2 rows for better layout)
    frm_btn = ttk.Frame(content_frame)
    frm_btn.grid(row=0, column=0, sticky="ew", padx=2, pady=(2, 4))
    frm_btn.grid_columnconfigure((0, 1, 2, 3), weight=1)

    widgets["btn_open"] = ttk.Button(frm_btn, text="📂 Open Video", command=on_open_video)
    widgets["btn_export"] = ttk.Button(frm_btn, text="💾 Export Video", command=on_export_video)
    widgets["btn_optimize"] = ttk.Button(frm_btn, text="🎯 Optimize Center", command=on_optimize_center)
    widgets["btn_reset"] = ttk.Button(frm_btn, text="🔁 Reset Defaults", command=on_reset)

    widgets["btn_open"].grid(row=0, column=0, padx=2, sticky="ew")
    widgets["btn_export"].grid(row=0, column=1, padx=2, sticky="ew")
    widgets["btn_optimize"].grid(row=0, column=2, padx=2, sticky="ew")
    widgets["btn_reset"].grid(row=0, column=3, padx=2, sticky="ew")
    
    widgets["btn_exit"] = ttk.Button(frm_btn, text="🚪 Exit", command=on_exit)
    widgets["btn_exit"].grid(row=1, column=0, columnspan=4, padx=2, pady=(2, 0), sticky="ew")

    # Parameters frame
    frm_params = ttk.LabelFrame(content_frame, text="Processing Parameters")
    frm_params.grid(row=2, column=0, sticky="ew", padx=2, pady=3, ipadx=3, ipady=2)
    frm_params.grid_columnconfigure(0, weight=0, minsize=130)
    frm_params.grid_columnconfigure(1, weight=1, minsize=260)
    frm_params.grid_columnconfigure(2, weight=0, minsize=56)

    def add_slider(row, label, key, from_, to_):
        ttk.Label(frm_params, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=1)
        var = tk.IntVar(value=int(params[key]))
        scale = ttk.Scale(
            frm_params,
            from_=from_,
            to=to_,
            orient="horizontal",
            length=260,
            command=lambda _v, k=key, _var=var: _var.set(int(float(_v))),
        )
        scale.set(params[key])
        scale.grid(row=row, column=1, sticky="ew", padx=4, pady=1)
        lbl_val = ttk.Label(frm_params, text=str(var.get()), width=6, anchor="e")
        lbl_val.grid(row=row, column=2, padx=(0, 4), sticky="e")

        def on_var(*_):
            v = int(var.get())
            if key == "blur":
                if v < 1:
                    v = 1
                if v % 2 == 0:
                    v += 1
                var.set(v)
            params[key] = v
            lbl_val.config(text=str(v))

        var.trace_add("write", on_var)
        widgets[f"scale_{key}"] = scale
        gui_vars_numeric[key] = var

    def add_slider_float(row, label, key, from_, to_):
        ttk.Label(frm_params, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=1)
        var = tk.IntVar(value=int(params[key] * 100))
        scale = ttk.Scale(
            frm_params,
            from_=from_,
            to=to_,
            orient="horizontal",
            length=260,
            command=lambda _v, k=key, _var=var: _var.set(int(float(_v))),
        )
        scale.set(int(params[key] * 100))
        scale.grid(row=row, column=1, sticky="ew", padx=4, pady=1)
        lbl_val = ttk.Label(frm_params, text=f"{params[key]*100:.0f}%", width=6, anchor="e")
        lbl_val.grid(row=row, column=2, padx=(0, 4), sticky="e")

        def on_var(*_):
            params[key] = max(0.0, min(1.0, float(var.get()) / 100.0))
            lbl_val.config(text=f"{params[key]*100:.0f}%")

        var.trace_add("write", on_var)
        widgets[f"scale_{key}"] = scale
        gui_vars_numeric[key] = var

    def add_slider_smin(row):
        key = "Smin"
        ttk.Label(frm_params, text="Min Pair Score (×100)").grid(row=row, column=0, sticky="w", padx=4, pady=1)
        var = tk.IntVar(value=int(params[key] * 100))
        scale = ttk.Scale(frm_params, from_=10, to=200, orient="horizontal", command=lambda _v, _var=var: _var.set(int(float(_v))))
        scale.set(int(params[key] * 100))
        lbl_val = ttk.Label(frm_params, text=f"{params[key]*100:.0f}", width=6, anchor="e")
        lbl_val.grid(row=row, column=2, padx=(0, 4), sticky="e")

        def on_var(*_):
            params[key] = max(0.1, min(2.0, float(var.get()) / 100.0))
            lbl_val.config(text=f"{params[key]*100:.0f}")

        var.trace_add("write", on_var)
        scale.grid(row=row, column=1, sticky="ew", padx=4, pady=1)
        widgets[f"scale_{key}"] = scale
        gui_vars_numeric[key] = var

    def add_slider_contrast(row):
        key = "contrast"
        ttk.Label(frm_params, text="Contrast (%)").grid(row=row, column=0, sticky="w", padx=4, pady=1)
        var = tk.IntVar(value=int(params[key]))
        scale = ttk.Scale(frm_params, from_=0, to=200, orient="horizontal", command=lambda _v, _var=var: _var.set(int(float(_v))))
        scale.set(int(params[key]))
        lbl_val = ttk.Label(frm_params, text=f"{params[key]}%", width=6, anchor="e")
        lbl_val.grid(row=row, column=2, padx=(0, 4), sticky="e")

        def on_var(*_):
            params[key] = max(0, min(200, int(var.get())))
            lbl_val.config(text=f"{params[key]}%")

        var.trace_add("write", on_var)
        scale.grid(row=row, column=1, sticky="ew", padx=4, pady=1)
        widgets[f"scale_{key}"] = scale
        gui_vars_numeric[key] = var

    row = 0
    add_slider(row, "Binary Threshold", "threshold", 0, 255); row += 1
    add_slider(row, "Gaussian Blur Size (odd)", "blur", 1, 25); row += 1
    add_slider_contrast(row); row += 1
    add_slider(row, "Min Blob Area (px²)", "minArea", 0, 200); row += 1
    add_slider(row, "Max Blob Area (px²)", "maxArea", 100, 200); row += 1
    add_slider(row, "Max Blob Width/Height (px)", "maxW", 1, 200); row += 1
    add_slider(row, "Max Radial Gap (px)", "maxRadGap", 0, 200); row += 1
    add_slider(row, "Max Angle Diff (deg)", "maxDMR", 0, 30); row += 1
    add_slider(row, "Max Center Offset (px)", "maxCenterOff", 0, 200); row += 1
    add_slider_float(row, "Weight: Angle Similarity", "w_theta", 0, 100); row += 1
    add_slider_float(row, "Weight: Area Similarity", "w_area", 0, 100); row += 1
    add_slider_float(row, "Weight: Center Alignment", "w_center", 0, 100); row += 1
    add_slider_smin(row); row += 1

    # Tracking parameters
    def add_slider_float_track(row, label, key, from_, to_):
        ttk.Label(frm_params, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=1)
        var = tk.DoubleVar(value=float(params[key]))
        scale = ttk.Scale(
            frm_params,
            from_=from_,
            to=to_,
            orient="horizontal",
            length=260,
            command=lambda _v, k=key, _var=var: _var.set(float(_v)),
        )
        scale.set(float(params[key]))
        scale.grid(row=row, column=1, sticky="ew", padx=4, pady=1)
        lbl_val = ttk.Label(frm_params, text=f"{params[key]:.1f}", width=6, anchor="e")
        lbl_val.grid(row=row, column=2, padx=(0, 4), sticky="e")

        def on_var(*_):
            params[key] = max(float(from_), min(float(to_), float(var.get())))
            lbl_val.config(text=f"{params[key]:.1f}")

        var.trace_add("write", on_var)
        widgets[f"scale_{key}"] = scale
        gui_vars_numeric[key] = var

    add_slider_float_track(row, "Track: Max Match Distance (px)", "track_max_match_dist", 5.0, 100.0); row += 1
    add_slider(row, "Track: Max Misses Before Retire", "track_max_misses", 1, 30); row += 1

    # Pairing method
    frm_method = ttk.LabelFrame(content_frame, text="Pairing Method")
    frm_method.grid(row=3, column=0, sticky="ew", padx=2, pady=(2, 1), ipadx=3, ipady=2)
    frm_method.grid_columnconfigure(0, weight=0)
    frm_method.grid_columnconfigure(1, weight=1)

    methods_display = [
        "Greedy (local best)",
        "Symmetric (mutual best)",
        "Hungarian (global best)",
    ]
    internal_from_display = {
        "Greedy (local best)": "Greedy",
        "Symmetric (mutual best)": "Symmetric",
        "Hungarian (global best)": "Hungarian",
    }
    desc_for_internal = {
        "Greedy": "Fast local matches; may miss optimal global pairing.",
        "Symmetric": "Pairs are mutual best to each other; moderate compute.",
        "Hungarian": "Global optimum across all candidates; most robust.",
    }

    ttk.Label(frm_method, text="Method").grid(row=0, column=0, sticky="w", padx=4, pady=2)
    current_disp = {
        "Greedy": methods_display[0],
        "Symmetric": methods_display[1],
        "Hungarian": methods_display[2],
    }.get(params.get("pair_method", "Hungarian"), methods_display[2])
    method_var = tk.StringVar(value=current_disp)
    cmb = ttk.Combobox(frm_method, values=methods_display, state="readonly", textvariable=method_var)
    cmb.grid(row=0, column=1, sticky="ew", padx=4, pady=2)
    lbl_desc = ttk.Label(frm_method, text=desc_for_internal.get(params.get("pair_method", "Hungarian"), ""))
    lbl_desc.grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 2))

    def on_method_change(*_):
        disp = method_var.get()
        internal = internal_from_display.get(disp, "Hungarian")
        params["pair_method"] = internal
        lbl_desc.config(text=desc_for_internal.get(internal, ""))

    cmb.bind("<<ComboboxSelected>>", on_method_change)
    method_var.trace_add("write", on_method_change)
    widgets["cmb_pair_method"] = cmb

    # Overlay Targets
    frm_target = ttk.LabelFrame(content_frame, text="Overlay Targets")
    frm_target.grid(row=4, column=0, sticky="ew", padx=2, pady=(2, 1), ipadx=3, ipady=2)
    frm_target.grid_columnconfigure((0, 1), weight=1)

    def add_target_check(col, text, key):
        var = tk.IntVar(value=int(overlay_targets[key]))
        chk = ttk.Checkbutton(
            frm_target, text=text, variable=var, command=lambda k=key, v=var: overlay_targets.__setitem__(k, int(v.get()))
        )
        chk.grid(row=0, column=col, padx=4, pady=2, sticky="w")
        widgets[f"chk_{key}"] = chk

    add_target_check(0, "Enable Tracked", "enable_tracked")
    add_target_check(1, "Enable Binary", "enable_binary")

    # Overlays
    frm_ov = ttk.LabelFrame(content_frame, text="Overlays")
    frm_ov.grid(row=5, column=0, sticky="ew", padx=2, pady=3, ipadx=3, ipady=2)
    for i in range(3):
        frm_ov.grid_columnconfigure(i, weight=1)

    def add_check(grid_r, grid_c, text, key):
        var = tk.IntVar(value=int(overlays[key]))
        chk = ttk.Checkbutton(
            frm_ov, text=text, variable=var, command=lambda k=key, v=var: overlays.__setitem__(k, int(v.get()))
        )
        chk.grid(row=grid_r, column=grid_c, padx=4, pady=2, sticky="w")
        gui_vars_check[key] = var
        widgets[f"chk_{key}"] = chk

    add_check(0, 0, "Blobs", "show_blobs")
    add_check(0, 1, "Center", "show_center")
    add_check(0, 2, "Pair Center", "show_pair_center")
    add_check(1, 0, "Lines", "show_lines")
    add_check(1, 1, "Rays", "show_rays")
    
    # Label mode dropdown
    ttk.Label(frm_ov, text="Color:").grid(row=2, column=0, padx=4, pady=2, sticky="w")
    label_mode_var = tk.StringVar(value=overlays.get("label_mode", "Red/Blue"))
    cmb_label = ttk.Combobox(
        frm_ov,
        values=["None", "Red/Blue", "Random"],
        state="readonly",
        textvariable=label_mode_var,
        width=12,
    )
    cmb_label.grid(row=2, column=1, padx=4, pady=2, sticky="w")
    cmb_label.set(overlays.get("label_mode", "Red/Blue"))
    
    def on_label_mode_change(*_):
        overlays["label_mode"] = label_mode_var.get()
    
    cmb_label.bind("<<ComboboxSelected>>", on_label_mode_change)
    label_mode_var.trace_add("write", on_label_mode_change)
    widgets["cmb_label_mode"] = cmb_label
    
    # Text labels checkbox
    add_check(2, 2, "Text Labels (#A/#B)", "show_text_labels")

    # Preview Overlay section
    frm_preview_ov = ttk.LabelFrame(content_frame, text="Preview Overlay")
    frm_preview_ov.grid(row=6, column=0, sticky="ew", padx=2, pady=3, ipadx=3, ipady=2)
    for i in range(2):
        frm_preview_ov.grid_columnconfigure(i, weight=1)

    def add_preview_check(grid_r, grid_c, text, key):
        var = tk.IntVar(value=int(overlays.get(key, 0)))
        chk = ttk.Checkbutton(
            frm_preview_ov, text=text, variable=var, command=lambda k=key, v=var: overlays.__setitem__(k, int(v.get()))
        )
        chk.grid(row=grid_r, column=grid_c, padx=4, pady=2, sticky="w")
        gui_vars_check[key] = var
        widgets[f"chk_{key}"] = chk

    add_preview_check(0, 0, "Current Stats", "show_current_stats")
    add_preview_check(0, 1, "Total Stats", "show_total_stats")

    ttk.Label(
        content_frame,
        text="Tip: Click in the 'Tracked' window to set optical center. ESC closes preview windows.",
    ).grid(row=7, column=0, sticky="w", padx=4, pady=(4, 2))
    
    # Configure content frame column to expand
    content_frame.grid_columnconfigure(0, weight=1)

    return root, widgets, gui_vars_numeric, gui_vars_check


def reset_defaults_ui(
    params: dict,
    overlays: dict,
    DEFAULT_PARAMS: dict,
    DEFAULT_OVERLAYS: dict,
    gui_vars_numeric: dict,
    gui_vars_check: dict,
    widgets: dict,
):
    # Reset model values
    params.clear(); params.update(DEFAULT_PARAMS)
    overlays.clear(); overlays.update(DEFAULT_OVERLAYS)

    # Reflect in GUI numeric sliders
    for key, var in gui_vars_numeric.items():
        try:
            var.set(params[key])
        except Exception:
            pass
    # Reflect in GUI checkboxes
    for key, var in gui_vars_check.items():
        try:
            var.set(overlays[key])
        except Exception:
            pass

    # Update pairing method combobox text if present
    try:
        default_disp = {
            "Greedy": "Greedy (local best)",
            "Symmetric": "Symmetric (mutual best)",
            "Hungarian": "Hungarian (global best)",
        }.get(params.get("pair_method", "Hungarian"), "Hungarian (global best)")
        cmb = widgets.get("cmb_pair_method")
        if cmb is not None:
            cmb.set(default_disp)
    except Exception:
        pass

    # Update label mode combobox if present
    try:
        cmb_label = widgets.get("cmb_label_mode")
        if cmb_label is not None:
            cmb_label.set(overlays.get("label_mode", "Red/Blue"))
    except Exception:
        pass


