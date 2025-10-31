#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UVC Camera GUI — Auto Exposure Only
-----------------------------------
✓ Always uses camera's built-in auto exposure
✓ Select capture format (YUYV or MJPG)
✓ Live adjustable brightness, contrast, saturation, gain
✓ Power line frequency dropdown
✓ MP4 recording (VLC playable)
✓ Live preview with debug console info
"""

import cv2, time, platform, subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# ---------- Utilities ----------
def safe_run(cmd):
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=2)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] v4l2-ctl returned {e.returncode}: {e.output.decode(errors='ignore').strip()}")
    except Exception as e:
        print(f"[WARN] safe_run error: {e}")

def tooltip(widget, text):
    tip=tk.Toplevel(widget); tip.withdraw(); tip.overrideredirect(True)
    lbl=tk.Label(tip,text=text,bg="#ffffe0",relief="solid",borderwidth=1,font=("TkDefaultFont",9))
    lbl.pack()
    def show(_): tip.geometry(f"+{widget.winfo_rootx()+30}+{widget.winfo_rooty()+10}"); tip.deiconify()
    def hide(_): tip.withdraw()
    widget.bind("<Enter>",show); widget.bind("<Leave>",hide)

# ---------- Camera ----------
class Camera:
    def __init__(self,index=0,fps=15,fourcc="YUYV"):
        self.index,self.fps,self.fourcc=index,fps,fourcc
        self.cap=None; self.device=f"/dev/video{index}"
        self._linux=platform.system().lower()=="linux"
        self.w,self.h=None,None

    def open(self):
        self.cap=cv2.VideoCapture(self.index,cv2.CAP_V4L2)
        if not self.cap.isOpened(): return False
        self.cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*self.fourcc))
        if self.fourcc=="MJPG":
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.cap.set(cv2.CAP_PROP_FPS,self.fps)
        time.sleep(0.3)
        self.w=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camera opened: {self.fourcc} mode at {self.w}×{self.h} @ {self.fps} FPS")
        return True

    def read(self):
        ok,frame=self.cap.read() if self.cap else (False,None)
        return frame if ok else None

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap=None
            print("[INFO] Camera released")

    def set_ctrl(self,name,value):
        if not self._linux or not self.cap: return
        val=int(value)
        safe_run(["v4l2-ctl","-d",self.device,"-c",f"{name}={val}"])
        print(f"[DEBUG] Set {name} = {val}")

# ---------- GUI ----------
class CaptureApp:
    def __init__(self,root):
        self.root=root; self.root.title("UVC Capture Control — Auto Exposure")
        self.mode=tk.StringVar(value="YUYV")
        self.cam=None; self.preview_on=False; self.video_writer=None
        self.last_time=time.time(); self.fps_est=0
        self.scale_percent=tk.DoubleVar(value=100.0)
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        main=ttk.Frame(self.root); main.pack(fill=tk.BOTH,expand=True)
        self.canvas=tk.Canvas(main,bg="black"); self.canvas.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
        ctrl=ttk.Frame(main); ctrl.pack(side=tk.RIGHT,fill=tk.Y,padx=6,pady=6)

        ttk.Label(ctrl,text="Camera format").pack(anchor="w")
        fmt_box=ttk.Combobox(ctrl,values=["YUYV","MJPG"],textvariable=self.mode,width=6)
        fmt_box.pack(pady=2)
        tooltip(fmt_box,
                "YUYV = raw, fastest, limited to ~640×480.\n"
                "MJPG = compressed, allows 720p/1080p but higher CPU decode.")
        self.format_desc=tk.Label(ctrl,justify="left",bg="#eef",relief="groove")
        self.format_desc.pack(fill=tk.X,pady=2)
        self._update_format_desc()
        fmt_box.bind("<<ComboboxSelected>>",lambda e:self._update_format_desc())

        # Scale
        ttk.Label(ctrl,text="Output scale (%)").pack(anchor="w",pady=(5,0))
        e=ttk.Entry(ctrl,textvariable=self.scale_percent,width=6); e.pack()
        tooltip(e,"Resize output relative to native resolution (1–100 %)")
        ttk.Button(ctrl,text="Apply Scale",command=self.update_scale_info).pack(pady=3)
        self.scale_label=ttk.Label(ctrl,text="Scaled: —"); self.scale_label.pack()

        ttk.Separator(ctrl,orient="horizontal").pack(fill=tk.X,pady=6)
        self.param_frame=ttk.Frame(ctrl); self.param_frame.pack(fill=tk.X,pady=4)
        self.build_sliders()
        ttk.Separator(ctrl,orient="horizontal").pack(fill=tk.X,pady=6)
        self.build_dropdowns()

        ttk.Separator(ctrl,orient="horizontal").pack(fill=tk.X,pady=6)
        b=ttk.Frame(ctrl); b.pack()
        ttk.Button(b,text="Open Cam",command=self.open_camera).grid(row=0,column=0,padx=3)
        ttk.Button(b,text="Start Preview",command=self.start_preview).grid(row=0,column=1,padx=3)
        ttk.Button(b,text="Stop",command=self.stop_preview).grid(row=0,column=2,padx=3)
        ttk.Button(b,text="Capture Frame",command=self.capture_frame).grid(row=1,column=0,columnspan=3,pady=2)
        ttk.Button(b,text="Record MP4",command=self.toggle_record).grid(row=2,column=0,columnspan=3,pady=2)
        ttk.Button(b,text="Defaults",command=self.set_defaults).grid(row=3,column=0,columnspan=3,pady=2)

    # ------------------------------------------------------------------
    def build_sliders(self):
        params={
            "brightness":("Offset −64–64",-64,64,0),
            "contrast":("Contrast 0–64",0,64,32),
            "saturation":("Color 0–128",0,128,60),
            "gain":("Gain 0–100",0,100,32),
        }
        self.num_vars={}
        for name,(desc,mn,mx,val) in params.items():
            ttk.Label(self.param_frame,text=name).pack(anchor="w")
            frame=tk.Frame(self.param_frame); frame.pack(fill=tk.X)
            var=tk.DoubleVar(value=val)
            ent=ttk.Entry(frame,textvariable=var,width=8)
            ent.pack(side=tk.RIGHT,padx=3)
            slider=tk.Scale(frame,from_=mn,to=mx,orient="horizontal",variable=var,resolution=1,length=150)
            slider.pack(side=tk.LEFT,fill=tk.X,expand=True)
            tooltip(ent,desc)
            def update(name=name,var=var):
                if self.cam:
                    self.cam.set_ctrl(name,var.get())
                    print(f"[DEBUG] {name} updated live to {var.get()}")
            var.trace_add("write",lambda *a,update=update:update())
            ent.bind("<Return>",lambda ev:update())
            self.num_vars[name]=var

    # ------------------------------------------------------------------
    def build_dropdowns(self):
        dropdowns={
            "power_line_frequency":{
                "label":"Power Line Frequency",
                "options":{0:"Disabled",1:"50 Hz",2:"60 Hz"},
                "default":2},
        }
        self.drop_vars={}
        for name,info in dropdowns.items():
            ttk.Label(self.param_frame,text=info["label"]).pack(anchor="w")
            var=tk.IntVar(value=info["default"])
            opt_names=[f"{k}: {v}" for k,v in info["options"].items()]
            combo=ttk.Combobox(self.param_frame,values=opt_names,width=14)
            combo.pack(pady=2)
            combo.set(f"{info['default']}: {info['options'][info['default']]}")
            def update(name=name,var=var,combo=combo):
                try:
                    sel=int(combo.get().split(":")[0])
                    var.set(sel)
                    if self.cam:
                        self.cam.set_ctrl(name,sel)
                        print(f"[DEBUG] {name} set to {sel} ({info['options'][sel]})")
                except Exception as e:
                    print(f"[WARN] Dropdown parse error {e}")
            combo.bind("<<ComboboxSelected>>",lambda e:update())
            tooltip(combo,"Select mode for "+info["label"])
            self.drop_vars[name]=var

    # ------------------------------------------------------------------
    def _update_format_desc(self):
        if self.mode.get()=="MJPG":
            txt=("MJPG: JPEG-compressed frames.\n"
                 "• Pros : supports 720p/1080p, lower USB load.\n"
                 "• Cons : slight latency, compression artifacts.")
        else:
            txt=("YUYV: uncompressed 4:2:2 stream.\n"
                 "• Pros : fast preview, low latency.\n"
                 "• Cons : limited to VGA on USB 2.0.")
        self.format_desc.config(text=txt)

    # ------------------------------------------------------------------
    def open_camera(self):
        if self.cam: self.cam.release()
        self.cam=Camera(fourcc=self.mode.get())
        if not self.cam.open():
            messagebox.showerror("Camera","Open failed")
            return
        self.update_scale_info()
        print("[INFO] Using automatic exposure (default)")

    def update_scale_info(self):
        if not self.cam: return
        p=max(1,min(100,float(self.scale_percent.get())))
        sw,sh=int(self.cam.w*p/100),int(self.cam.h*p/100)
        self.scale_label.config(text=f"Scaled: {sw}×{sh}")
        print(f"[INFO] {self.mode.get()} {self.cam.w}×{self.cam.h} → {sw}×{sh} ({p:.1f} %)")

    def scaled_size(self):
        p=max(1,min(100,float(self.scale_percent.get())))
        return int(self.cam.w*p/100),int(self.cam.h*p/100)

    def start_preview(self):
        if not self.cam: self.open_camera()
        if not self.preview_on:
            print("[INFO] Starting preview")
            self.preview_on=True
            self.last_time=time.time()
            self._update_frame()

    def stop_preview(self):
        if self.preview_on:
            print("[INFO] Stopping preview")
        self.preview_on=False
        if self.video_writer:
            print("[INFO] Recording stopped")
            self.video_writer.release()
            self.video_writer=None

    def set_defaults(self):
        if not self.cam: return
        defaults={"brightness":0,"contrast":32,"saturation":60,"gain":32}
        for k,v in defaults.items():
            if k in self.num_vars:self.num_vars[k].set(v)
            self.cam.set_ctrl(k,v)
        print("[INFO] Defaults applied")

    def capture_frame(self):
        if not self.cam:return
        frame=self.cam.read()
        if frame is None:return
        w,h=self.scaled_size(); frame=cv2.resize(frame,(w,h))
        ts=time.strftime("%Y%m%d_%H%M%S"); name=f"frame_{w}x{h}_{ts}.png"
        cv2.imwrite(name,frame)
        print("[INFO] Saved",name)

    def toggle_record(self):
        if not self.cam:return
        if self.video_writer:
            self.video_writer.release(); self.video_writer=None
            print("[INFO] Recording stopped");return
        ts=time.strftime("%Y%m%d_%H%M%S")
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"); w,h=self.scaled_size()
        name=f"video_{w}x{h}_{ts}.mp4"
        self.video_writer=cv2.VideoWriter(name,fourcc,self.cam.fps,(w,h))
        print("[INFO] Recording started:",name)

    def _update_frame(self):
        if not self.preview_on or not self.cam:
            return
        frame = self.cam.read()
        if frame is not None:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_YUY2)

            w, h = self.scaled_size()
            rgb_resized = cv2.resize(rgb, (w, h))

            # --- FPS estimation ---
            now = time.time()
            dt = now - self.last_time
            self.last_time = now
            if dt > 0:
                self.fps_est = 0.9 * self.fps_est + 0.1 * (1 / dt)

            # --- Display text only in the preview ---
            rgb_preview = rgb_resized.copy()
            cv2.putText(rgb_preview,
                        f"{self.mode.get()} {self.cam.w}x{self.cam.h}  FPS:{self.fps_est:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # --- Update GUI preview ---
            im = Image.fromarray(rgb_preview)
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            scale = min(cw / w, ch / h)
            im = im.resize((int(w * scale), int(h * scale)), Image.NEAREST)
            imgtk = ImageTk.PhotoImage(image=im)
            self.canvas.create_image(cw // 2, ch // 2, image=imgtk)
            self.canvas.image = imgtk

            # --- Save clean frame for recording ---
            if self.video_writer:
                self.video_writer.write(cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR))

        self.root.after(int(1000 / (self.cam.fps or 15)), self._update_frame)


    def on_close(self):
        print("[INFO] Closing application")
        if self.cam:self.cam.release()
        self.stop_preview(); self.root.destroy()

# ---------- Main ----------
def main():
    root=tk.Tk(); app=CaptureApp(root)
    root.protocol("WM_DELETE_WINDOW",app.on_close)
    root.mainloop()

if __name__=="__main__": main()
