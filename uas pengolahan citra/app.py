import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import image_processor  # Modul buatan kita

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Pengolahan Citra Digital (UAS)")
        self.root.geometry("1280x800")
        self.root.configure(bg="#121212")
        
        # Inisialisasi variabel gambar
        self.orig_img = None      # CV2 BGR Image
        self.processed_img = None # CV2 BGR/Grayscale/Binary Image
        self.current_filename = ""
        self.dataset_dir = "dataset"
        self.webcam_active = False
        self.cap = None
        
        # Konfigurasi Gaya (Styling)
        self.setup_styles()
        
        # Membuat Tata Letak (Layout)
        self.create_widgets()
        
        # Muat daftar dataset gambar
        self.refresh_dataset_list()
        
        # Muat gambar pertama jika ada
        self.load_default_image()
        
        # Penanganan saat window ditutup
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.webcam_active = False
        if self.cap is not None:
            self.cap.release()
        plt.close('all') # Tutup plot matplotlib
        self.root.destroy()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Warna Palette Modern Dark Mode
        self.BG_COLOR = "#121212"
        self.CARD_BG = "#1E1E1E"
        self.ACCENT_COLOR = "#00ADB5"
        self.TEXT_COLOR = "#EEEEEE"
        self.TEXT_MUTED = "#888888"
        self.BORDER_COLOR = "#393E46"
        
        # Konfigurasi widget umum
        self.style.configure('.', background=self.BG_COLOR, foreground=self.TEXT_COLOR)
        self.style.configure('TFrame', background=self.BG_COLOR)
        self.style.configure('Card.TFrame', background=self.CARD_BG, relief='flat')
        self.style.configure('TLabel', background=self.BG_COLOR, foreground=self.TEXT_COLOR, font=('Segoe UI', 10))
        self.style.configure('Card.TLabel', background=self.CARD_BG, foreground=self.TEXT_COLOR, font=('Segoe UI', 10))
        self.style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'), foreground=self.ACCENT_COLOR, background=self.BG_COLOR)
        self.style.configure('Subheader.TLabel', font=('Segoe UI', 12, 'bold'), foreground=self.ACCENT_COLOR, background=self.CARD_BG)
        self.style.configure('Title.TLabel', font=('Segoe UI', 11, 'bold'), foreground=self.TEXT_COLOR, background=self.CARD_BG)
        self.style.configure('Desc.TLabel', font=('Segoe UI', 9), foreground=self.TEXT_MUTED, background=self.CARD_BG)
        
        # Desain tombol flat yang modern
        self.style.configure('TButton', background=self.BORDER_COLOR, foreground=self.TEXT_COLOR, 
                             font=('Segoe UI', 10, 'bold'), borderwidth=0, padding=6)
        self.style.map('TButton', 
                       background=[('active', self.ACCENT_COLOR), ('pressed', '#00979E')],
                       foreground=[('active', '#121212'), ('pressed', '#121212')])
        
        self.style.configure('Accent.TButton', background=self.ACCENT_COLOR, foreground='#121212', 
                             font=('Segoe UI', 10, 'bold'), borderwidth=0, padding=6)
        self.style.map('Accent.TButton', 
                       background=[('active', '#00FFF5'), ('pressed', '#00979E')])

        self.style.configure('TCombobox', fieldbackground=self.CARD_BG, background=self.BORDER_COLOR, 
                             foreground=self.TEXT_COLOR, borderwidth=0, padding=4)
        self.style.map('TCombobox', 
                       fieldbackground=[('readonly', self.CARD_BG)],
                       foreground=[('readonly', self.TEXT_COLOR)])

    def create_widgets(self):
        # 1. Header Frame
        header_frame = ttk.Frame(self.root, padding=(15, 10, 15, 5))
        header_frame.pack(fill='x', side='top')
        
        lbl_title = ttk.Label(header_frame, text="PENGOLAHAN CITRA DIGITAL - MINI PROJECT UAS", style="Header.TLabel")
        lbl_title.pack(side='left')
        
        lbl_subtitle = ttk.Label(header_frame, text="Teknik Informatika - Universitas Pelita Bangsa", style="TLabel")
        lbl_subtitle.pack(side='right', pady=5)
        
        # Garis pemisah tipis
        sep = ttk.Separator(self.root, orient='horizontal')
        sep.pack(fill='x', padx=15, pady=5)
        
        # 2. Main Container
        main_container = ttk.Frame(self.root, padding=10)
        main_container.pack(fill='both', expand=True)
        
        # Sidebar Kiri (Dataset & File Controls)
        left_sidebar = ttk.Frame(main_container, width=220)
        left_sidebar.pack(fill='y', side='left', padx=(5, 10))
        left_sidebar.pack_propagate(False)
        
        # Card untuk File Manager
        fm_card = ttk.Frame(left_sidebar, style="Card.TFrame", padding=10)
        fm_card.pack(fill='both', expand=True)
        
        lbl_fm = ttk.Label(fm_card, text="Dataset Gambar", style="Subheader.TLabel")
        lbl_fm.pack(anchor='w', pady=(0, 10))
        
        # Listbox untuk Gambar Dataset
        self.dataset_listbox = tk.Listbox(fm_card, bg="#121212", fg=self.TEXT_COLOR, 
                                          selectbackground=self.ACCENT_COLOR, selectforeground="#121212",
                                          font=('Segoe UI', 9), bd=0, highlightthickness=1, 
                                          highlightbackground=self.BORDER_COLOR)
        self.dataset_listbox.pack(fill='both', expand=True, pady=(0, 10))
        self.dataset_listbox.bind('<<ListboxSelect>>', self.on_dataset_select)
        
        # Tombol File Controls
        btn_open = ttk.Button(fm_card, text="Buka File Kustom", command=self.open_custom_file)
        btn_open.pack(fill='x', pady=4)
        
        self.btn_webcam = ttk.Button(fm_card, text="Gunakan Kamera", command=self.toggle_webcam)
        self.btn_webcam.pack(fill='x', pady=4)
        
        btn_refresh = ttk.Button(fm_card, text="Segarkan Dataset", command=self.refresh_dataset_list)
        btn_refresh.pack(fill='x', pady=4)

        # Middle Panel (Gambar Original, Hasil, dan Histogram)
        middle_panel = ttk.Frame(main_container)
        middle_panel.pack(fill='both', expand=True, side='left', padx=5)
        
        # Row 1: Preview Gambar (Original vs Processed)
        preview_frame = ttk.Frame(middle_panel)
        preview_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Frame Gambar Original
        self.orig_card = ttk.Frame(preview_frame, style="Card.TFrame", padding=5)
        self.orig_card.pack(fill='both', expand=True, side='left', padx=(0, 5))
        
        lbl_orig_title = ttk.Label(self.orig_card, text="Gambar Asli", style="Title.TLabel")
        lbl_orig_title.pack(anchor='w', padx=5, pady=2)
        
        self.lbl_orig_canvas = ttk.Label(self.orig_card, text="Tidak ada gambar dimuat", anchor='center', background="#121212")
        self.lbl_orig_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Frame Gambar Hasil
        self.proc_card = ttk.Frame(preview_frame, style="Card.TFrame", padding=5)
        self.proc_card.pack(fill='both', expand=True, side='left', padx=(5, 0))
        
        lbl_proc_title = ttk.Label(self.proc_card, text="Gambar Hasil Pemrosesan", style="Title.TLabel")
        lbl_proc_title.pack(anchor='w', padx=5, pady=2)
        
        self.lbl_proc_canvas = ttk.Label(self.proc_card, text="Hasil akan muncul di sini", anchor='center', background="#121212")
        self.lbl_proc_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Row 2: Histogram
        hist_card = ttk.Frame(middle_panel, style="Card.TFrame", padding=10)
        hist_card.pack(fill='x', side='bottom')
        
        lbl_hist_title = ttk.Label(hist_card, text="Analisis Citra (Histogram Frekuensi Warna)", style="Subheader.TLabel")
        lbl_hist_title.pack(anchor='w', pady=(0, 5))
        
        # Plot Matplotlib untuk Histogram
        self.fig, self.ax = plt.subplots(figsize=(6, 2.2), facecolor='#1E1E1E')
        self.ax.set_facecolor('#121212')
        self.ax.tick_params(colors='#EEEEEE', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('#393E46')
        
        self.hist_canvas = FigureCanvasTkAgg(self.fig, master=hist_card)
        self.hist_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Sidebar Kanan (Controls Algoritma & Parameter)
        right_sidebar = ttk.Frame(main_container, width=280)
        right_sidebar.pack(fill='y', side='right', padx=(10, 5))
        right_sidebar.pack_propagate(False)
        
        ctrl_card = ttk.Frame(right_sidebar, style="Card.TFrame", padding=12)
        ctrl_card.pack(fill='both', expand=True)
        
        lbl_ctrl = ttk.Label(ctrl_card, text="Metode Pengolahan", style="Subheader.TLabel")
        lbl_ctrl.pack(anchor='w', pady=(0, 8))
        
        # Dropdown Kategori Algoritma
        lbl_cat = ttk.Label(ctrl_card, text="Kategori Proses:", style="Card.TLabel")
        lbl_cat.pack(anchor='w', pady=(5, 2))
        
        self.categories = [
            "1. Konversi Citra",
            "2. Perbaikan Kualitas",
            "3. Filtering Noise",
            "4. Deteksi Tepi (Edge)",
            "5. Segmentasi Citra",
            "6. Deteksi Wajah (Bonus)"
        ]
        self.cat_combobox = ttk.Combobox(ctrl_card, values=self.categories, state="readonly")
        self.cat_combobox.pack(fill='x', pady=(0, 10))
        self.cat_combobox.current(0)
        self.cat_combobox.bind("<<ComboboxSelected>>", self.on_category_change)
        
        # Dropdown Operasi Spesifik
        lbl_op = ttk.Label(ctrl_card, text="Operasi Spesifik:", style="Card.TLabel")
        lbl_op.pack(anchor='w', pady=(5, 2))
        
        self.op_combobox = ttk.Combobox(ctrl_card, state="readonly")
        self.op_combobox.pack(fill='x', pady=(0, 15))
        self.op_combobox.bind("<<ComboboxSelected>>", self.on_operation_change)
        
        # Frame Parameter Dinamis (akan diisi slider secara dinamis)
        self.param_outer_frame = ttk.LabelFrame(ctrl_card, text=" Pengaturan Parameter ", padding=10)
        self.param_outer_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        self.param_frame = ttk.Frame(self.param_outer_frame)
        self.param_frame.pack(fill='both', expand=True)
        
        # Tombol Aksi di bagian Bawah
        btn_save = ttk.Button(ctrl_card, text="Simpan Gambar Hasil", style="Accent.TButton", command=self.save_processed_image)
        btn_save.pack(fill='x', pady=5)
        
        btn_batch = ttk.Button(ctrl_card, text="Proses Masal (Batch 20)", command=self.run_batch_processing)
        btn_batch.pack(fill='x', pady=5)
        
        self.lbl_status = ttk.Label(ctrl_card, text="Status: Siap.", style="Desc.TLabel")
        self.lbl_status.pack(anchor='w', pady=(10, 0))

        # Inisialisasi dropdown operasi
        self.on_category_change(None)

    # --- PENGENDALIAN KATEGORI & OPERASI ---
    def on_category_change(self, event):
        cat = self.cat_combobox.get()
        if "1. Konversi" in cat:
            ops = ["Grayscale", "Biner"]
        elif "2. Perbaikan" in cat:
            ops = ["Histogram Equalization", "Contrast Stretching", "Kecerahan (Brightness)", "Penajaman (Sharpening)"]
        elif "3. Filtering" in cat:
            ops = ["Mean Filter", "Median Filter", "Gaussian Filter"]
        elif "4. Deteksi" in cat:
            ops = ["Sobel Edge", "Canny Edge", "Prewitt Edge"]
        elif "5. Segmentasi" in cat:
            ops = ["Manual Thresholding", "Otsu Thresholding", "K-Means Clustering", "Watershed Segmentation"]
        elif "6. Deteksi Wajah" in cat:
            ops = ["Deteksi Wajah Haar Cascade"]
        else:
            ops = []
            
        self.op_combobox.config(values=ops)
        if ops:
            self.op_combobox.current(0)
            self.on_operation_change(None)

    def on_operation_change(self, event):
        # Bersihkan widget parameter yang ada
        for widget in self.param_frame.winfo_children():
            widget.destroy()
            
        op = self.op_combobox.get()
        self.sliders = {} # Untuk menyimpan referensi nilai slider
        
        # Buat slider/dropdown parameter secara dinamis sesuai operasi
        if op == "Grayscale":
            self.add_description("Mengubah citra warna RGB menjadi citra berderajat keabuan (Grayscale). Tidak membutuhkan parameter tambahan.")
            
        elif op == "Biner":
            self.add_description("Mengubah citra ke format biner (hitam-putih) berdasarkan nilai ambang batas (threshold) manual.")
            self.add_slider("Threshold", 0, 255, 127)
            
        elif op == "Histogram Equalization":
            self.add_description("Meningkatkan kontras citra secara otomatis dengan meratakan distribusi histogram frekuensi intensitas warna.")
            
        elif op == "Contrast Stretching":
            self.add_description("Meningkatkan kontras citra secara linear dengan memetakan nilai piksel ke rentang dinamis penuh.")
            self.add_slider("Persentil Bawah (Min)", 0, 10, 2)
            self.add_slider("Persentil Atas (Max)", 90, 100, 98)
            
        elif op == "Kecerahan (Brightness)":
            self.add_description("Mengatur kecerahan gambar dengan menambahkan/mengurangi konstanta pada intensitas piksel.")
            self.add_slider("Kecerahan", -100, 100, 0)
            
        elif op == "Penajaman (Sharpening)":
            self.add_description("Mempertajam tepi citra menggunakan metode Unsharp Masking (menambah frekuensi tinggi kembali ke citra asli).")
            self.add_slider("Kekuatan (Strength)", 0.0, 3.0, 1.0, resolution=0.1)
            
        elif op in ["Mean Filter", "Median Filter"]:
            self.add_description(f"Filter penapis berbasis rata-rata/median untuk menghaluskan gambar dan mereduksi derau (noise).")
            self.add_slider("Ukuran Kernel", 3, 15, 3, step=2)  # Hanya ganjil
            
        elif op == "Gaussian Filter":
            self.add_description("Filter penghalusan berbasis distribusi Gauss. Memberikan blur yang lebih alami.")
            self.add_slider("Ukuran Kernel", 3, 15, 3, step=2)
            self.add_slider("Sigma (Standar Deviasi)", 0.1, 5.0, 1.0, resolution=0.1)
            
        elif op == "Sobel Edge":
            self.add_description("Deteksi tepi menggunakan operator gradien Sobel pada arah horizontal (X) dan vertikal (Y).")
            self.add_slider("Ukuran Kernel", 1, 7, 3, step=2)
            
        elif op == "Canny Edge":
            self.add_description("Deteksi tepi Canny yang menggunakan proses multi-tahap (noise reduction, gradient calculation, non-maximum suppression, hysteresis thresholding).")
            self.add_slider("Threshold 1 (Bawah)", 0, 255, 50)
            self.add_slider("Threshold 2 (Atas)", 0, 255, 150)
            
        elif op == "Prewitt Edge":
            self.add_description("Deteksi tepi menggunakan operator gradien Prewitt dengan konvolusi manual.")
            
        elif op == "Manual Thresholding":
            self.add_description("Segmentasi objek dengan binarisasi manual. Piksel di atas threshold menjadi putih, lainnya hitam.")
            self.add_slider("Threshold", 0, 255, 127)
            
        elif op == "Otsu Thresholding":
            self.add_description("Segmentasi objek menggunakan nilai ambang batas otomatis yang optimal secara statistik berdasarkan varians histogram.")
            
        elif op == "K-Means Clustering":
            self.add_description("Segmentasi citra berbasis pengelompokan warna (unsupervised ML). Mengelompokkan warna ke dalam K cluster.")
            self.add_slider("Jumlah Cluster (K)", 2, 8, 3)
            
        elif op == "Watershed Segmentation":
            self.add_description("Segmentasi berbasis kontur dengan analogi 'banjir air' dari daerah penanda (markers) menggunakan transformasi jarak.")
            
        elif op == "Deteksi Wajah Haar Cascade":
            self.add_description("Bonus UAS: Mendeteksi wajah secara real-time / statis menggunakan algoritma Haar Cascade Classifier.")
            self.add_slider("Scale Factor", 1.05, 1.5, 1.1, resolution=0.05)
            self.add_slider("Min Neighbors", 1, 10, 5)

        # Picu pemrosesan awal setelah parameter diubah
        self.apply_processing()

    # --- WIDGET HELPER ---
    def add_description(self, text):
        lbl = ttk.Label(self.param_frame, text=text, style="Desc.TLabel", wraplength=230, justify="left")
        lbl.pack(fill='x', pady=(0, 10))

    def add_slider(self, label, min_val, max_val, init_val, step=None, resolution=None):
        frame = ttk.Frame(self.param_frame, style="Card.TFrame")
        frame.pack(fill='x', pady=5)
        
        lbl_val = ttk.Label(frame, text=f"{label}: {init_val}", style="Card.TLabel")
        lbl_val.pack(anchor='w')
        
        var = tk.DoubleVar(value=init_val)
        self.sliders[label] = var
        
        def on_slider_move(val):
            # Format display value
            float_val = float(val)
            if step is not None:
                # Memaksa step (misal: ganjil saja)
                rounded_val = int(round(float_val))
                if step == 2 and rounded_val % 2 == 0:
                    rounded_val += 1
                var.set(rounded_val)
                float_val = rounded_val
            
            if resolution is None and step is None:
                lbl_val.config(text=f"{label}: {int(round(float_val))}")
            elif resolution is not None and resolution < 1:
                lbl_val.config(text=f"{label}: {float_val:.2f}")
            else:
                lbl_val.config(text=f"{label}: {int(float_val)}")
            
            # Jalankan pemrosesan citra secara real-time
            self.apply_processing()
            
        slider = tk.Scale(frame, from_=min_val, to_=max_val, variable=var, orient='horizontal',
                          command=on_slider_move, bg=self.CARD_BG, fg=self.TEXT_COLOR,
                          troughcolor=self.BORDER_COLOR, activebackground=self.ACCENT_COLOR,
                          highlightthickness=0, bd=0)
        
        # Handling resolusi slider desimal
        if resolution is not None:
            slider.config(resolution=resolution)
            
        slider.pack(fill='x', pady=(2, 0))

    # --- MANAJEMEN GAMBAR ---
    def refresh_dataset_list(self):
        self.dataset_listbox.delete(0, tk.END)
        if os.path.exists(self.dataset_dir):
            files = [f for f in os.listdir(self.dataset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for file in sorted(files):
                self.dataset_listbox.insert(tk.END, file)
            self.lbl_status.config(text=f"Dataset: memuat {len(files)} gambar.")
        else:
            self.lbl_status.config(text="Folder 'dataset' tidak ditemukan.")

    def load_default_image(self):
        if self.dataset_listbox.size() > 0:
            self.dataset_listbox.selection_set(0)
            self.on_dataset_select(None)

    def on_dataset_select(self, event):
        selection = self.dataset_listbox.curselection()
        if selection:
            filename = self.dataset_listbox.get(selection[0])
            filepath = os.path.join(self.dataset_dir, filename)
            self.current_filename = filename
            self.load_image(filepath)

    def open_custom_file(self):
        # Hentikan kamera jika sedang berjalan
        if self.webcam_active:
            self.toggle_webcam()
            
        filepath = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if filepath:
            self.current_filename = os.path.basename(filepath)
            self.load_image(filepath)

    def load_image(self, filepath):
        self.orig_img = cv2.imread(filepath)
        if self.orig_img is None:
            messagebox.showerror("Error", f"Gagal memuat gambar: {filepath}")
            return
        
        self.display_image(self.orig_img, self.lbl_orig_canvas)
        self.apply_processing()

    # --- DETEKSI WEBCAM (KAMERA REAL-TIME) ---
    def toggle_webcam(self):
        if self.webcam_active:
            # Hentikan Webcam
            self.webcam_active = False
            self.btn_webcam.config(text="Gunakan Kamera", style="TButton")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.lbl_status.config(text="Kamera dihentikan.")
            # Kembalikan ke gambar awal
            self.load_default_image()
        else:
            # Mulai Webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Kamera Error", "Gagal membuka kamera / webcam tidak terdeteksi.")
                self.cap = None
                return
                
            self.webcam_active = True
            self.btn_webcam.config(text="Hentikan Kamera", style="Accent.TButton")
            self.lbl_status.config(text="Kamera aktif secara real-time.")
            self.webcam_loop()

    def webcam_loop(self):
        if self.webcam_active and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Balik frame horizontal agar seperti cermin
                frame = cv2.flip(frame, 1)
                self.orig_img = frame.copy()
                
                # Tampilkan gambar asli kamera
                self.display_image(self.orig_img, self.lbl_orig_canvas)
                
                # Jalankan pemrosesan citra
                self.apply_processing()
                
            # Lakukan rekursi loop frame setiap 10ms
            self.root.after(10, self.webcam_loop)

    # --- PROSES CITRA & UPDATE DISPLAY ---
    def apply_processing(self):
        if self.orig_img is None:
            return
            
        op = self.op_combobox.get()
        img = self.orig_img.copy()
        
        try:
            # 1. Konversi Citra
            if op == "Grayscale":
                self.processed_img = image_processor.rgb_to_grayscale(img)
            elif op == "Biner":
                thresh = int(self.sliders["Threshold"].get())
                self.processed_img = image_processor.rgb_to_binary(img, thresh)
                
            # 2. Perbaikan Kualitas Citra
            elif op == "Histogram Equalization":
                self.processed_img = image_processor.histogram_equalization(img)
            elif op == "Contrast Stretching":
                low = int(self.sliders["Persentil Bawah (Min)"].get())
                high = int(self.sliders["Persentil Atas (Max)"].get())
                self.processed_img = image_processor.contrast_stretching(img, low, high)
            elif op == "Kecerahan (Brightness)":
                val = int(self.sliders["Kecerahan"].get())
                self.processed_img = image_processor.adjust_brightness(img, val)
            elif op == "Penajaman (Sharpening)":
                strength = float(self.sliders["Kekuatan (Strength)"].get())
                self.processed_img = image_processor.sharpen_image(img, strength)
                
            # 3. Filtering
            elif op == "Mean Filter":
                ksize = int(self.sliders["Ukuran Kernel"].get())
                self.processed_img = image_processor.mean_filter(img, ksize)
            elif op == "Median Filter":
                ksize = int(self.sliders["Ukuran Kernel"].get())
                self.processed_img = image_processor.median_filter(img, ksize)
            elif op == "Gaussian Filter":
                ksize = int(self.sliders["Ukuran Kernel"].get())
                sigma = float(self.sliders["Sigma (Standar Deviasi)"].get())
                self.processed_img = image_processor.gaussian_filter(img, ksize, sigma)
                
            # 4. Deteksi Tepi
            elif op == "Sobel Edge":
                ksize = int(self.sliders["Ukuran Kernel"].get())
                self.processed_img = image_processor.sobel_edge(img, ksize)
            elif op == "Canny Edge":
                t1 = int(self.sliders["Threshold 1 (Bawah)"].get())
                t2 = int(self.sliders["Threshold 2 (Atas)"].get())
                self.processed_img = image_processor.canny_edge(img, t1, t2)
            elif op == "Prewitt Edge":
                self.processed_img = image_processor.prewitt_edge(img)
                
            # 5. Segmentasi Citra
            elif op == "Manual Thresholding":
                thresh = int(self.sliders["Threshold"].get())
                self.processed_img = image_processor.threshold_segmentation(img, thresh, "Manual")
            elif op == "Otsu Thresholding":
                self.processed_img = image_processor.threshold_segmentation(img, method="Otsu")
            elif op == "K-Means Clustering":
                k = int(self.sliders["Jumlah Cluster (K)"].get())
                self.processed_img = image_processor.kmeans_segmentation(img, k)
            elif op == "Watershed Segmentation":
                self.processed_img = image_processor.watershed_segmentation(img)
                
            # 6. Bonus Deteksi Wajah
            elif op == "Deteksi Wajah Haar Cascade":
                sf = float(self.sliders["Scale Factor"].get())
                mn = int(self.sliders["Min Neighbors"].get())
                self.processed_img, face_count = image_processor.detect_faces(img, sf, mn)
                if not self.webcam_active:
                    self.lbl_status.config(text=f"Deteksi Wajah: Menemukan {face_count} wajah.")
            
            # Tampilkan Gambar Hasil
            self.display_image(self.processed_img, self.lbl_proc_canvas)
            
            # Update Histogram Citra Hasil secara real-time
            self.update_histogram(self.processed_img)
            
        except Exception as e:
            # Tampilkan error di status tanpa pop-up untuk kenyamanan
            self.lbl_status.config(text=f"Error pemrosesan: {str(e)[:40]}")

    def display_image(self, img_cv, label_widget):
        # Konversi BGR OpenCV ke RGB PIL
        if len(img_cv.shape) == 3:
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        else:
            # Grayscale / Binary
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
            
        pil_img = Image.fromarray(img_rgb)
        
        # Dapatkan dimensi frame display label
        label_w = label_widget.winfo_width()
        label_h = label_widget.winfo_height()
        
        # Berikan nilai default jika label belum selesai me-layout (misal saat startup)
        if label_w <= 1 or label_h <= 1:
            label_w = 400
            label_h = 300
            
        # Pertahankan Aspek Rasio Gambar
        img_w, img_h = pil_img.size
        ratio = min(label_w / img_w, label_h / img_h)
        new_w = max(1, int(img_w * ratio))
        new_h = max(1, int(img_h * ratio))
        
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Konversi ke Tkinter PhotoImage
        tk_img = ImageTk.PhotoImage(pil_img)
        label_widget.config(image=tk_img, text="")
        label_widget.image = tk_img  # Menghindari garbage collection

    # --- PLOT HISTOGRAM MATPLOTLIB ---
    def update_histogram(self, img_cv):
        self.ax.clear()
        self.ax.set_facecolor('#121212')
        self.ax.set_xlim([0, 256])
        self.ax.set_ylim(auto=True)
        self.ax.tick_params(colors='#EEEEEE', labelsize=8)
        
        # Matikan garis tepi grafik luar
        for spine in self.ax.spines.values():
            spine.set_color('#393E46')

        if len(img_cv.shape) == 3:
            # Citra Berwarna (Plot Red, Green, Blue)
            colors = ('b', 'g', 'r')  # OpenCV menggunakan urutan BGR
            plot_colors = ('#007ACC', '#2ECC71', '#E74C3C') # Hex warna modern
            
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img_cv], [i], None, [256], [0, 256])
                self.ax.plot(hist, color=plot_colors[i], alpha=0.8, linewidth=1.5)
        else:
            # Citra Grayscale / Biner (Plot Tunggal)
            hist = cv2.calcHist([img_cv], [0], None, [256], [0, 256])
            self.ax.plot(hist, color=self.ACCENT_COLOR, alpha=0.9, linewidth=1.5)
            self.ax.fill_between(range(256), hist.flatten(), color=self.ACCENT_COLOR, alpha=0.2)
            
        self.ax.grid(True, color="#2D2D2D", linestyle="--", linewidth=0.5)
        self.hist_canvas.draw()

    # --- SIMPAN & BATCH PROCESSING ---
    def save_processed_image(self):
        if self.processed_img is None:
            messagebox.showwarning("Peringatan", "Tidak ada gambar hasil yang bisa disimpan.")
            return
            
        out_dir = "processed_output"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        base_name = os.path.splitext(self.current_filename)[0] if self.current_filename else "kamera"
        op = self.op_combobox.get().lower().replace(" ", "_").replace("(", "").replace(")", "")
        filepath = os.path.join(out_dir, f"{base_name}_{op}.jpg")
        
        # Simpan menggunakan OpenCV
        cv2.imwrite(filepath, self.processed_img)
        messagebox.showinfo("Berhasil", f"Gambar disimpan di:\n{filepath}")
        self.lbl_status.config(text=f"Berhasil menyimpan ke: {filepath}")

    def run_batch_processing(self):
        if not os.path.exists(self.dataset_dir) or len(os.listdir(self.dataset_dir)) == 0:
            messagebox.showwarning("Peringatan", f"Dataset tidak ditemukan di folder '{self.dataset_dir}'.")
            return
            
        op = self.op_combobox.get()
        
        # Mulai proses masal di Thread terpisah agar GUI tidak hang/beku
        thread = threading.Thread(target=self.batch_processing_worker, args=(op,))
        thread.start()

    def batch_processing_worker(self, op):
        out_dir = "batch_output"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        files = [f for f in os.listdir(self.dataset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_files = len(files)
        
        if total_files == 0:
            return
            
        self.lbl_status.config(text="Proses Masal: Sedang memproses...")
        self.root.title("Aplikasi Pengolahan Citra Digital (PROSES MASAL SEDANG BERJALAN...)")
        
        processed_count = 0
        for i, file in enumerate(files, 1):
            filepath = os.path.join(self.dataset_dir, file)
            img = cv2.imread(filepath)
            if img is None:
                continue
                
            try:
                # Salin parameter dinamis yang aktif dari thread utama
                # Gunakan pemrosesan yang sama persis
                if op == "Grayscale":
                    res = image_processor.rgb_to_grayscale(img)
                elif op == "Biner":
                    thresh = int(self.sliders["Threshold"].get())
                    res = image_processor.rgb_to_binary(img, thresh)
                elif op == "Histogram Equalization":
                    res = image_processor.histogram_equalization(img)
                elif op == "Contrast Stretching":
                    low = int(self.sliders["Persentil Bawah (Min)"].get())
                    high = int(self.sliders["Persentil Atas (Max)"].get())
                    res = image_processor.contrast_stretching(img, low, high)
                elif op == "Kecerahan (Brightness)":
                    val = int(self.sliders["Kecerahan"].get())
                    res = image_processor.adjust_brightness(img, val)
                elif op == "Penajaman (Sharpening)":
                    strength = float(self.sliders["Kekuatan (Strength)"].get())
                    res = image_processor.sharpen_image(img, strength)
                elif op == "Mean Filter":
                    ksize = int(self.sliders["Ukuran Kernel"].get())
                    res = image_processor.mean_filter(img, ksize)
                elif op == "Median Filter":
                    ksize = int(self.sliders["Ukuran Kernel"].get())
                    res = image_processor.median_filter(img, ksize)
                elif op == "Gaussian Filter":
                    ksize = int(self.sliders["Ukuran Kernel"].get())
                    sigma = float(self.sliders["Sigma (Standar Deviasi)"].get())
                    res = image_processor.gaussian_filter(img, ksize, sigma)
                elif op == "Sobel Edge":
                    ksize = int(self.sliders["Ukuran Kernel"].get())
                    res = image_processor.sobel_edge(img, ksize)
                elif op == "Canny Edge":
                    t1 = int(self.sliders["Threshold 1 (Bawah)"].get())
                    t2 = int(self.sliders["Threshold 2 (Atas)"].get())
                    res = image_processor.canny_edge(img, t1, t2)
                elif op == "Prewitt Edge":
                    res = image_processor.prewitt_edge(img)
                elif op == "Manual Thresholding":
                    thresh = int(self.sliders["Threshold"].get())
                    res = image_processor.threshold_segmentation(img, thresh, "Manual")
                elif op == "Otsu Thresholding":
                    res = image_processor.threshold_segmentation(img, method="Otsu")
                elif op == "K-Means Clustering":
                    k = int(self.sliders["Jumlah Cluster (K)"].get())
                    res = image_processor.kmeans_segmentation(img, k)
                elif op == "Watershed Segmentation":
                    res = image_processor.watershed_segmentation(img)
                elif op == "Deteksi Wajah Haar Cascade":
                    sf = float(self.sliders["Scale Factor"].get())
                    mn = int(self.sliders["Min Neighbors"].get())
                    res, _ = image_processor.detect_faces(img, sf, mn)
                else:
                    res = img
                
                # Simpan gambar hasil
                op_name = op.lower().replace(" ", "_").replace("(", "").replace(")", "")
                base_name = os.path.splitext(file)[0]
                out_path = os.path.join(out_dir, f"{base_name}_{op_name}.jpg")
                cv2.imwrite(out_path, res)
                processed_count += 1
                
                # Update status
                self.lbl_status.config(text=f"Proses Masal: Memproses {i}/{total_files}...")
            except Exception as e:
                print(f"Error memproses berkas {file}: {e}")
                
        # Akhir dari proses masal
        self.root.title("Aplikasi Pengolahan Citra Digital (UAS)")
        self.lbl_status.config(text=f"Selesai! {processed_count} gambar disimpan di 'batch_output'.")
        
        # Pop-up pemberitahuan
        messagebox.showinfo("Proses Masal Selesai", 
                            f"Berhasil memproses {processed_count} gambar menggunakan metode '{op}'.\n\nHasil disimpan di folder 'batch_output/'")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    # Jalankan penyesuaian ukuran awal canvas setelah window dibuat sempurna
    root.update()
    # Panggil ulang pemrosesan agar canvas terhitung dengan pas aspek rasionya
    app.on_operation_change(None)
    root.mainloop()
