import tkinter as tk
from tkinter import filedialog, ttk
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

class MedicalImageViewer:
    def __init__(self, master):
        self.master = master
        self.master.title("Advanced DICOM Viewer - Robust Windowing")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.volume = None
        self.header_info = {}
        
        self.setup_ui()

    def setup_ui(self):
        self.top_frame = tk.Frame(self.master)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        tk.Button(self.top_frame, text="DICOMフォルダを開く", command=self.load_dicom_folder).pack(side=tk.LEFT)
        self.info_label = tk.Label(self.top_frame, text="フォルダを選択してください", justify=tk.LEFT)
        self.info_label.pack(side=tk.LEFT, padx=20)

        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.patch.set_facecolor('#222222') # 背景を暗くして見やすく
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.ctrl_frame = tk.Frame(self.master)
        self.ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.sliders = {}
        # ラベル, 初期値, 範囲(min, max)
        controls = [
            ("Axial", 0, (0, 100)), ("Coronal", 0, (0, 100)), ("Sagittal", 0, (0, 100)),
            ("WW", 400, (1, 3000)), ("WL", 40, (-1000, 2000))
        ]

        for label, default, (r_min, r_max) in controls:
            frame = tk.Frame(self.ctrl_frame)
            frame.pack(side=tk.LEFT, padx=10)
            tk.Label(frame, text=label).pack()
            s = tk.Scale(frame, from_=r_min, to=r_max, orient=tk.HORIZONTAL, length=150,
                         command=lambda x: self.update_plots())
            s.set(default)
            s.pack()
            self.sliders[label] = s

    def robust_wl_ww(self, vol):
        p1, p99 = np.percentile(vol, [1, 99])
        wl = float((p1 + p99) / 2.0)
        ww = float(max(p99 - p1, 1.0))
        return wl, ww

    def wlww_to_uint8(self, slice_data, wl, ww):
        low = wl - ww / 2.0
        high = wl + ww / 2.0
        sl = slice_data.astype(np.float32)
        sl = (sl - low) / max(high - low, 1.0)
        sl = np.clip(sl, 0.0, 1.0)
        return (sl * 255.0).astype(np.uint8)

    def load_dicom_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path: return

        files = [pydicom.dcmread(os.path.join(folder_path, f)) 
                 for f in os.listdir(folder_path) if f.lower().endswith('.dcm')]
        if not files: return
        files.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # HU値への正規化
        self.volume = np.stack([f.pixel_array.astype(np.float32) * float(getattr(f, 'RescaleSlope', 1)) + 
                                float(getattr(f, 'RescaleIntercept', 0)) for f in files])

        # 自動WL/WWの設定
        auto_wl, auto_ww = self.robust_wl_ww(self.volume)
        self.sliders["WL"].set(int(auto_wl))
        self.sliders["WW"].set(int(auto_ww))

        ref = files[0]
        thick = getattr(ref, 'SliceThickness', 1.0)
        spacing = getattr(ref, 'PixelSpacing', [1.0, 1.0])
        
        self.info_label.config(text=f"サイズ: {ref.Rows}x{ref.Columns} | 厚み: {thick}mm | 数: {len(files)}")

        self.sliders["Axial"].config(to=self.volume.shape[0]-1)
        self.sliders["Coronal"].config(to=self.volume.shape[1]-1)
        self.sliders["Sagittal"].config(to=self.volume.shape[2]-1)
        
        self.asp_axial = spacing[0] / spacing[1]
        self.asp_coronal = thick / spacing[1]
        self.asp_sagittal = thick / spacing[0]

        self.update_plots()

    def update_plots(self):
        if self.volume is None: return

        z, y, x = [self.sliders[k].get() for k in ["Axial", "Coronal", "Sagittal"]]
        wl, ww = self.sliders["WL"].get(), self.sliders["WW"].get()

        color_ax, color_cor, color_sag = 'cyan', 'lime', 'red'

        # --- 修正箇所：Axialを上下反転のみに変更 ---
        # [::-1, :] は行方向（上下）のみを反転させるスライス処理
        axial_img = self.volume[z, ::-1, :] 
        coronal_img = self.volume[:, y, :]
        sagittal_img = self.volume[:, :, x]

        # --- 修正箇所：スライス線の位置計算 ---
        max_y = self.volume.shape[1] - 1
        # Axialが上下反転しているため、Coronal線(水平線)の位置を反転させる
        # Sagittal線(垂直線)は左右反転していないため、元のxをそのまま使う
        ax_line_h = x 
        ax_line_v = max_y - y
        # ---------------------------------------

        data = [
            (axial_img, "Axial", color_sag, color_cor, ax_line_h, ax_line_v, self.asp_axial, z, color_ax),
            (coronal_img, "Coronal", color_sag, color_ax, x, z, self.asp_coronal, y, color_cor),
            (sagittal_img, "Sagittal", color_cor, color_ax, y, z, self.asp_sagittal, x, color_sag)
        ]
        
        # (以下の描画ループは変更なし)

        for i, (img, title, cv, ch, lh, lv, asp, idx, t_color) in enumerate(data):
            self.axes[i].clear()
            # 引用した正規化関数を適用
            win_img = self.wlww_to_uint8(img, wl, ww)
            self.axes[i].imshow(win_img, cmap='gray', origin='lower', aspect=asp)
            self.axes[i].set_title(f"{title}\nSlice: {idx}", color=t_color, fontweight='bold')
            self.axes[i].axvline(lh, color=cv, lw=1.5, alpha=0.8)
            self.axes[i].axhline(lv, color=ch, lw=1.5, alpha=0.8)
            self.axes[i].axis('off')

        self.canvas.draw()

    def on_closing(self):
        plt.close('all')
        self.master.quit()
        self.master.destroy()
        sys.exit()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1400x800")
    app = MedicalImageViewer(root)
    root.mainloop()
