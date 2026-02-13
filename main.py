import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import numpy as np
import os
import cv2
import mask_array

class ImageTab:
    def __init__(self, parent_notebook, img_path=None, pil_image=None):
        self.notebook = parent_notebook
        self.img_path = img_path

        if pil_image is not None:
            self.original = pil_image
        else:
            self.original = Image.open(img_path)

        arr = np.array(self.original)
        if self.original.mode in ("RGB", "RGBA") and arr.ndim == 3:
            if np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2]):
                self.original = self.original.convert("L")

        self.display = self.original.copy()

        filename = os.path.basename(img_path) if img_path else "Nowy obraz"
        self.frame = ttk.Frame(self.notebook)
        self.notebook.add(self.frame, text=filename)

        self.canvas = tk.Canvas(self.frame, bg="#DADADA")
        self.canvas.pack(fill="both", expand=True)

        self.zoom_factor = 1.0

        self.render()
        self.canvas.bind("<Configure>", lambda e: self.render())


    def render(self):
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 10 or canvas_h < 10:
            return

        # Skalowanie
        new_w = int(self.original.width * self.zoom_factor)
        new_h = int(self.original.height * self.zoom_factor)
        self.display = self.original.resize((new_w, new_h))

        self.tk_img = ImageTk.PhotoImage(self.display)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.tk_img, anchor="center")

    def set_zoom(self, factor):
        self.zoom_factor = factor
        self.render()

    def zoom_fit(self):
        # Dopasowanie do okna
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        scale_w = cw / self.original.width
        scale_h = ch / self.original.height
        self.zoom_factor = min(scale_w, scale_h)
        self.render()

    def zoom_original(self):
        self.zoom_factor = 1.0
        self.render()

#  Główna Aplikacja
class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikacja do przetwarzania obrazów")
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        self._tab_objects = {}
        self.create_menu()

    #  Obsługa Menu
    def create_menu(self):
        menubar = tk.Menu(self.root)

        # PLiK
        file_menu = tk.Menu(menubar, tearoff=0)

        file_menu.add_command(label="Otwórz obraz", command=self.open_image)
        file_menu.add_command(label="Zapisz jako...", command=self.save_as_image)
        file_menu.add_command(label="Duplikuj obraz", command=self.duplicate_image)
        file_menu.add_separator()
        file_menu.add_command(label="Zamknij kartę", command=self.close_tab)
        file_menu.add_command(label="Zamknij wszystkie karty", command=self.close_all_tabs)
        file_menu.add_separator()
        file_menu.add_command(label="Zakończ", command=self.root.quit)

        menubar.add_cascade(label="Plik", menu=file_menu)

        # WIDOK
        view_menu = tk.Menu(menubar, tearoff=0)

        view_menu.add_command(label="Skala 1:1", command=self.zoom_original)
        view_menu.add_command(label="Dopasuj do okna", command=self.zoom_fit)
        view_menu.add_command(label="Pełny ekran", command=self.toggle_fullscreen)

        menubar.add_cascade(label="Widok", menu=view_menu)

        # LABORATORIUM 1
        lab1_menu = tk.Menu(menubar, tearoff=0)

        # LUT
        lab1_menu.add_command(label="LUT", command=self.show_lut)

        # HISTOGRAM
        hist_menu = tk.Menu(lab1_menu, tearoff=0)

        hist_menu.add_command(label="Rysuj histogram", command=self.show_histogram)
        hist_menu.add_command(label="Rozciąganie liniowe", command=self.linear_stretch)
        hist_menu.add_command(label="Rozciąganie z przesyceniem 5%", command=self.linear_stretch_clipped)
        hist_menu.add_command(label="Equalizacja histogramu", command=self.hist_equalization)

        lab1_menu.add_cascade(label="Histogram", menu=hist_menu)

        # OPERACJE PUNKTOWE
        op_menu = tk.Menu(lab1_menu, tearoff=0)

        op_menu.add_command(label="Negacja", command=self.op_negation)
        op_menu.add_command(label="Kwantyzacja", command=self.op_quantization)
        op_menu.add_command(label="Progowanie binarne", command=self.op_threshold_binary)
        op_menu.add_command(label="Progowanie z zachowaniem szarości", command=self.op_threshold_grayscale)
        lab1_menu.add_cascade(label="Operacje punktowe", menu=op_menu)

        menubar.add_cascade(label="Laboratorium 1", menu=lab1_menu)

        # LABORATORIUM 2
        lab2_menu = tk.Menu(menubar, tearoff=0)

        # OPERACJE WIELOARGUMENTOWE
        multi_menu = tk.Menu(lab2_menu, tearoff=0)
        multi_menu.add_command(label="Dodawanie obrazów (bez wysycenia)", command=lambda: self.multi_image_operation("add_no_sat"))
        multi_menu.add_command(label="Dodawanie obrazów (z wysyceniem)", command=lambda: self.multi_image_operation("add_sat"))
        multi_menu.add_separator()
        multi_menu.add_command(label="Różnica bezwzględna obrazów", command=lambda: self.multi_image_operation("abs_diff"))

        # OPERACJE Z LICZBĄ
        num_menu = tk.Menu(lab2_menu, tearoff=0)
        num_menu.add_command(label="Dodawanie liczby (bez wysycenia)", command=lambda: self.single_image_number_operation("add_no_sat"))
        num_menu.add_command(label="Dodawanie liczby (z wysyceniem)", command=lambda: self.single_image_number_operation("add_sat"))
        num_menu.add_separator()
        num_menu.add_command(label="Mnożenie przez liczbę (bez wysycenia)", command=lambda: self.single_image_number_operation("mul_no_sat"))
        num_menu.add_command(label="Mnożenie przez liczbę (z wysyceniem)", command=lambda: self.single_image_number_operation("mul_sat"))
        num_menu.add_separator()
        num_menu.add_command(label="Dzielenie przez liczbę", command=lambda: self.single_image_number_operation("div"))

        lab2_menu.add_cascade(label="Operacje na wielu obrazach", menu=multi_menu)

        lab2_menu.add_cascade(label="Operacje z liczbą", menu=num_menu)

        # OPERACJE BINARNE
        logic_menu = tk.Menu(lab2_menu, tearoff=0)

        logic_menu.add_command(label="NOT", command=self.logic_not)
        logic_menu.add_command(label="AND", command=lambda: self.logic_binary("AND"))
        logic_menu.add_command(label="OR", command=lambda: self.logic_binary("OR"))
        logic_menu.add_command(label="XOR", command=lambda: self.logic_binary("XOR"))
        logic_menu.add_separator()
        logic_menu.add_command(label="Konwersja: 8-bit → binarny", command=self.to_binary_mask)
        logic_menu.add_command(label="Konwersja: binarny → 8-bit", command=self.from_binary_mask)

        lab2_menu.add_cascade(label="Operacje logiczne", menu=logic_menu)

        # FILTRY LINIOWE I DETEKCJA KRAWĘDZI
        filters_menu = tk.Menu(lab2_menu, tearoff=0)

        filters_menu.add_command(label="Uśrednianie", command=lambda: self.filter_select("mean"))
        filters_menu.add_command(label="Uśrednianie (wagowe)", command=lambda: self.filter_select("weighted"))
        filters_menu.add_command(label="Gauss", command=lambda: self.filter_select("gauss"))
        filters_menu.add_separator()
        filters_menu.add_command(label="Wyostrzanie (Laplace)", command=lambda: self.filter_select("laplace"))
        filters_menu.add_separator()
        filters_menu.add_command(label="Prewitt (8 kierunków)", command=lambda: self.filter_select("prewitt"))
        filters_menu.add_separator()
        filters_menu.add_command(label="Sobel (X/Y)", command=lambda: self.filter_select("sobel"))
        filters_menu.add_separator()
        filters_menu.add_command(label="Filtr medianowy", command=self.median_filter_select)
        filters_menu.add_separator()
        filters_menu.add_command(label="Detekcja krawędzi (Canny)", command=self.canny_select)

        lab2_menu.add_cascade(label="Filtry i krawędzie", menu=filters_menu)

        menubar.add_cascade(label="Laboratorium 2", menu=lab2_menu)

        # LABORATORIUM 3
        lab3_menu = tk.Menu(menubar, tearoff=0)

        # ROZCIĄGANIE HISTOGRAMU
        lab3_menu.add_command(label="Rozciąganie histogramu", command=self.stretch_popup)

        # SEGMENTACJA
        seg_menu = tk.Menu(lab3_menu, tearoff=0)

        seg_menu.add_command(label="Progowanie (dwa progi)", command=self.threshold_two_popup)
        seg_menu.add_command(label="Progowanie metodą Otsu", command=self.threshold_otsu)
        seg_menu.add_command(label="Progowanie adaptacyjne", command=self.threshold_adaptive_popup)

        lab3_menu.add_cascade(label="Segmentowanie", menu=seg_menu)

        # MORFOLOGIA MATEMATYCZNA
        morph_menu = tk.Menu(lab3_menu, tearoff=0)

        morph_menu.add_command(label="Erozja", command=lambda: self.morph_select("erode"))
        morph_menu.add_command(label="Dylacja", command=lambda: self.morph_select("dilate"))
        morph_menu.add_command(label="Otwarcie", command=lambda: self.morph_select("open"))
        morph_menu.add_command(label="Zamknięcie", command=lambda: self.morph_select("close"))

        lab3_menu.add_cascade(label="Morfologia", menu=morph_menu)

        # SZKIELETYZACJA
        lab3_menu.add_command(label="Szkieletyzacja", command=self.skeletonize)

        menubar.add_cascade(label="Laboratorium 3", menu=lab3_menu)

        # LABORATORIUM 4
        lab4_menu = tk.Menu(menubar, tearoff=0)

        # ANALIZA OBRAZU I ZAPIS
        lab4_menu.add_command(label="Analiza obiektu binarnego", command=self.analyze_binary_object)

        # HOUGH
        lab4_menu.add_command(label="Transformata Hougha",command=self.hough_lines_popup)

        menubar.add_cascade(label="Laboratorium 4", menu=lab4_menu)

        # MINIPROJEKT
        menubar.add_command(label="Mini-Projekt", command=self.advanced_histogram_transform)

        # INFO
        menubar.add_command(label="Informacje", command=self.show_about)

        self.root.config(menu=menubar)

    # FUNKCJE UŻYTKOWE
    def create_tab_from_array(self, array, operation_name, source_tab=None):
        img = Image.fromarray(array.astype(np.uint8))

        # Ustalanie nazwy
        if source_tab and source_tab.img_path:
            base = os.path.basename(source_tab.img_path)
            base = os.path.splitext(base)[0]
        else:
            base = "obraz"

        new_title = f"{base}_{operation_name}"

        # Utworzenie nowej karty z obrazem wynikowym
        new_tab = ImageTab(self.notebook, img_path=None, pil_image=img)
        tab_id = self.notebook.tabs()[-1]
        self._tab_objects[tab_id] = new_tab
        self.notebook.tab(tab_id, text=new_title)

        new_tab.img_path = None

        return new_tab

    def get_active_tab(self):
        tab_id = self.notebook.select()
        if not tab_id:
            return None

        return self._tab_objects.get(tab_id)

    def require_active_tab(self):
        tab = self.get_active_tab()
        if not tab:
            messagebox.showwarning("Błąd", "Brak aktywnej karty z obrazem.")
            return None
        return tab

    def require_grayscale(self, tab):
        if tab.original.mode != "L":
            messagebox.showwarning("Błąd", "Operacja dostępna tylko dla obrazów monochromatycznych (L).")
            return False
        return True

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Obrazy", "*.bmp *.tif *.png *.jpg"), ("Wszystkie pliki", "*.*")]
        )
        if not path:
            return

        tab = ImageTab(self.notebook, path)
        self._tab_objects[self.notebook.tabs()[-1]] = tab

    def save_as_image(self):
        tab = self.require_active_tab()
        if not tab:
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tif")
            ]
        )

        if not path:
            return

        root, ext = os.path.splitext(path)
        if ext == "":
            path = path + ".png"

        try:
            tab.original.save(path)
        except Exception as e:
            messagebox.showerror("Błąd zapisu", str(e))
            return

        tab.img_path = path
        messagebox.showinfo("Zapisz", "Zapisano obraz na dysku.")

    def duplicate_image(self):
        tab = self.require_active_tab()
        if not tab:
            return

        img = tab.original.copy()

        if tab.img_path:
            base = os.path.basename(tab.img_path)
            base = os.path.splitext(base)[0]
        else:
            base = "obraz"

        new_title = f"{base}_copy"

        new_tab = ImageTab(self.notebook, img_path=None, pil_image=img)
        tab_id = self.notebook.tabs()[-1]
        self._tab_objects[tab_id] = new_tab
        self.notebook.tab(tab_id, text=new_title)

        messagebox.showinfo("Duplikacja", "Zduplikowano obraz.")

    def close_tab(self):
        tab_id = self.notebook.select()
        if not tab_id:
            messagebox.showwarning("Błąd", "Brak aktywnej karty z obrazem.")
            return

        self._tab_objects.pop(tab_id, None)
        self.notebook.forget(tab_id)

    def close_all_tabs(self):
        if not self._tab_objects:
            messagebox.showwarning("Błąd", "Brak aktywnych kart z obrazami.")
            return

        for tab_id in list(self._tab_objects.keys()):
            try:
                self.notebook.forget(tab_id)
            except tk.TclError:
                pass

        self._tab_objects.clear()

        messagebox.showinfo("Zamknięcie kart", "Zamknięto wszystkie karty z obrazami.")

    # WIDOK
    def zoom_original(self):
        tab = self.get_active_tab()
        if tab:
            tab.zoom_original()

    def zoom_fit(self):
        tab = self.get_active_tab()
        if tab:
            tab.zoom_fit()

    def toggle_fullscreen(self):
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))

    # LABORATORIUM 1
    def show_lut(self):
        tab = self.require_active_tab()
        if not tab:
            return

        # Generowanie LUT
        img = np.array(tab.original)

        # MONO
        if tab.original.mode == "L":
            lut = np.bincount(img.flatten(), minlength=256)
            data = {"mode": "L", "lut": lut}

        # RGB
        elif tab.original.mode in ("RGB", "RGBA"):
            img = img[:, :, :3]
            hist_r = np.bincount(img[:, :, 0].flatten(), minlength=256)
            hist_g = np.bincount(img[:, :, 1].flatten(), minlength=256)
            hist_b = np.bincount(img[:, :, 2].flatten(), minlength=256)

            data = {"mode": "RGB", "R": hist_r, "G": hist_g, "B": hist_b}

        else:
            messagebox.showwarning("Nieobsługiwany format", tab.original.mode)
            return

        win = tk.Toplevel(self.root)
        win.title("Tablica LUT")
        win.geometry("800x600")

        tree = ttk.Treeview(win)
        tree.pack(fill="both", expand=True)

        if data["mode"] == "L":
            tree["columns"] = ("value", "count")
            tree.heading("value", text="Value")
            tree.heading("count", text="Count")
            tree.column("#0", width=0, stretch=False)

            for i in range(256):
                tree.insert("", "end", values=(i, data["lut"][i]))

        else:
            tree["columns"] = ("value", "R", "G", "B")
            tree.heading("value", text="Value")
            tree.heading("R", text="R")
            tree.heading("G", text="G")
            tree.heading("B", text="B")
            tree.column("#0", width=0, stretch=False)

            for i in range(256):
                tree.insert("", "end", values=(i, data["R"][i], data["G"][i], data["B"][i]))

    def _draw_hist_window(self, title, hist, color="black"):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("600x500")

        canvas = tk.Canvas(win, bg="white")
        canvas.pack(fill="both", expand=True)

        # Statystyki histogramu
        stats_box = tk.Text(win, height=6)
        stats_box.pack(fill="x")

        values = np.repeat(np.arange(256), hist)

        stats_box.insert("end", f"Min: {values.min()}\n")
        stats_box.insert("end", f"Max: {values.max()}\n")
        stats_box.insert("end", f"Średnia: {values.mean():.2f}\n")
        stats_box.insert("end", f"Odchylenie std.: {values.std():.2f}\n")
        stats_box.insert("end", f"Mediana: {np.median(values):.2f}\n")

        def draw():
            canvas.delete("all")

            w = canvas.winfo_width()
            h = canvas.winfo_height()
            margin_left = 10
            margin_bottom = 30

            hist_w = w
            hist_h = h - margin_bottom - 10

            max_val = max(hist) if max(hist) > 0 else 1
            bar_w = hist_w / 256

            canvas.create_line(
                margin_left, h - margin_bottom,
                             w - 10, h - margin_bottom,
                width=2
            )
            for x in range(0, 256, 32):
                xpos = margin_left + x * bar_w
                canvas.create_line(xpos, h - margin_bottom, xpos, h - margin_bottom + 5)
                canvas.create_text(xpos, h - margin_bottom + 15, text=str(x), font=("Arial", 8))

            for i in range(256):
                if hist[i] == 0:
                    continue
                bar_h = (hist[i] / max_val) * hist_h

                if bar_h < 1:
                    bar_h = 1

                x1 = margin_left + i * bar_w
                y1 = (h - margin_bottom) - bar_h
                x2 = margin_left + (i + 1) * bar_w
                y2 = h - margin_bottom

                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

        canvas.bind("<Configure>", lambda e: draw())

    def show_histogram(self):
        tab = self.require_active_tab()
        if not tab:
            return

        img = np.array(tab.original)

        # MONO
        if tab.original.mode == "L":
            hist = np.zeros(256, dtype=np.int64)
            for v in img.flatten():
                hist[v] += 1

            self._draw_hist_window("Histogram (L)", hist)

        else:
            # RGB
            img = img[:, :, :3]

            hist_r = np.zeros(256, dtype=np.int64)
            hist_g = np.zeros(256, dtype=np.int64)
            hist_b = np.zeros(256, dtype=np.int64)

            for ch, arr in zip(range(3), [hist_r, hist_g, hist_b]):
                for v in img[:, :, ch].flatten():
                    arr[v] += 1

            self._draw_hist_window("Histogram R", hist_r, color="red")
            self._draw_hist_window("Histogram G", hist_g, color="green")
            self._draw_hist_window("Histogram B", hist_b, color="blue")

    def linear_stretch(self):
        tab = self.require_active_tab()
        if not tab:
            return

        img = np.array(tab.original)

        if tab.original.mode == "L":
            min_v = img.min()
            max_v = img.max()
            out = (img - min_v) * (255 / (max_v - min_v))
            out = out.astype(np.uint8)
        else:
            out = img.astype(float)
            for ch in range(3):
                c = img[:, :, ch]
                min_v = c.min()
                max_v = c.max()
                out[:, :, ch] = (c - min_v) * (255 / (max_v - min_v))

        self.create_tab_from_array(out, "linear_stretch", source_tab=tab)
        messagebox.showinfo("Rozciąganie", "Rozciąganie histogramu zakończone pomyślnie.")

    def linear_stretch_clipped(self):
        tab = self.require_active_tab()
        if not tab:
            return

        img = np.array(tab.original)
        n = img.size
        cut = int(n * 0.025)

        if tab.original.mode == "L":
            flat = np.sort(img.flatten())
            min_v = flat[cut]
            max_v = flat[-cut]
            img = np.clip(img, min_v, max_v)
            out = (img - min_v) * (255 / (max_v - min_v))
        elif tab.original.mode in ("RGB", "RGBA"):
            out = img.astype(np.float32)
            for ch in range(3):
                c = img[:, :, ch]
                flat = np.sort(c.flatten())
                min_v = flat[cut]
                max_v = flat[-cut]
                c = np.clip(c, min_v, max_v)
                img[:, :, ch] = (c - min_v) * (255 / (max_v - min_v))
                out = img
        else:
            messagebox.showwarning("Błąd operacji", "Nieobsługiwany tryb obrazu.")
            return

        out = np.clip(out, 0, 255).astype(np.uint8)

        self.create_tab_from_array(out, "linear_stretch_clipped", source_tab=tab)
        messagebox.showinfo("Rozciąganie 5%", "Rozciąganie histogramu z przesyceniem zakończone pomyślnie.")

    def hist_equalization(self):
        tab = self.require_active_tab()
        if not tab:
            return

        img = np.array(tab.original)

        if tab.original.mode == "L":
            hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
            cdf = hist.cumsum()
            cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            cdf = cdf.astype(np.uint8)
            out = cdf[img]
            self.create_tab_from_array(out, "equalized", source_tab=tab)

        else:
            # RGB
            out = img.copy()
            for ch in range(3):
                channel = img[:, :, ch]
                hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
                cdf = hist.cumsum()
                cdf_min = cdf[np.nonzero(cdf)][0]
                cdf = (cdf - cdf_min) * 255 / (cdf[-1] - cdf_min)
                cdf = cdf.astype(np.uint8)
                out[:, :, ch] = cdf[channel]

            self.create_tab_from_array(out, "equalized", source_tab=tab)

        tab.render()
        messagebox.showinfo("Equalizacja", "Equalizacja zakończona pomyślnie.")

    def op_negation(self):
        tab = self.require_active_tab()
        if not tab:
            return

        if not self.require_grayscale(tab):
            return

        img = np.array(tab.original)
        out = 255 - img

        self.create_tab_from_array(out, "negation", source_tab=tab)
        messagebox.showinfo("Negacja", "Negacja zakończona pomyślnie.")

    def op_quantization(self):
        tab = self.require_active_tab()
        if not tab:
            return

        if not self.require_grayscale(tab):
            return

        # Kwantyzacja
        N = tk.simpledialog.askinteger("Kwantyzacja", "Podaj liczbę poziomów szarości:")
        if N < 2 or N > 256:
            messagebox.showwarning("Błąd", "Liczba poziomów musi być w zakresie 2–256.")
            return

        img = np.array(tab.original)
        step = 256 / N
        out = np.floor(img / step) * step
        out = out.astype(np.uint8)

        self.create_tab_from_array(out, "quantization", source_tab=tab)
        messagebox.showinfo("Kwantyzacja", "Kwantyzacja zakończona pomyślnie.")

    def op_threshold_binary(self):
        tab = self.require_active_tab()
        if not tab:
            return

        if not self.require_grayscale(tab):
            return

        t = tk.simpledialog.askinteger("Progowanie binarne", "Podaj próg (0–255):")
        if t is None:
            return

        img = np.array(tab.original)
        out = np.where(img >= t, 255, 0)

        self.create_tab_from_array(out, "binary_threshold", source_tab=tab)
        messagebox.showinfo("Progowanie", "Progowanie binarne zakończone pomyślnie.")

    def op_threshold_grayscale(self):
        tab = self.require_active_tab()
        if not tab:
            return

        if not self.require_grayscale(tab):
            return

        t = tk.simpledialog.askinteger("Progowanie z zachowaniem", "Podaj próg (0–255):")
        if t is None:
            return

        img = np.array(tab.original)
        out = img.copy()
        out[out < t] = 0

        self.create_tab_from_array(out, "grayscale_threshold", source_tab=tab)
        messagebox.showinfo("Progowanie", "Progowanie z zachowaniem poziomów szarości zakończono pomyślnie.")

    # LABORATORIUM 2
    def choose_images_popup(self):
        # Okno z listą obrazów do wybrania
        if not hasattr(self, "_tab_objects") or len(self._tab_objects) < 2:
            messagebox.showwarning("Brak danych", "Musisz mieć otwarte co najmniej 2 obrazy.")
            return None

        win = tk.Toplevel(self.root)
        win.title("Wybierz obrazy do operacji")
        win.geometry("300x300")

        vars_list = []

        for tab_id, tab in self._tab_objects.items():
            var = tk.BooleanVar()
            cb = tk.Checkbutton(win, text=self.notebook.tab(tab_id, "text"), variable=var)
            cb.pack(anchor="w")
            vars_list.append((var, tab))

        result = []

        def confirm():
            for var, tab in vars_list:
                if var.get():
                    result.append(tab)
            win.destroy()

        tk.Button(win, text="OK", command=confirm).pack()

        win.wait_window()
        return result

    def multi_image_operation(self, op_type):
        tabs = self.choose_images_popup()
        if not tabs:
            return
        if len(tabs) < 2:
            messagebox.showwarning("Błąd", "Musisz wybrać co najmniej dwa obrazy.")
            return

        # Weryfikacja typów i rozmiarów
        for t in tabs:
            if t.original.mode != "L":
                messagebox.showwarning("Błąd", "Wszystkie obrazy muszą być monochromatyczne (L).")
                return
        w, h = tabs[0].original.size
        for t in tabs:
            if t.original.size != (w, h):
                messagebox.showwarning("Błąd", "Wszystkie obrazy muszą mieć ten sam rozmiar.")
                return

        imgs = [np.array(t.original, dtype=np.float32) for t in tabs]

        # Operacje na wielu obrazach
        if op_type == "add_no_sat":
            max_sum = sum(im.max() for im in imgs)
            if max_sum > 255:
                scale = 255.0 / max_sum
                imgs_scaled = [im * scale for im in imgs]
            else:
                imgs_scaled = imgs
            out = sum(imgs_scaled)


        elif op_type == "add_sat":
            out = sum(imgs)
            out = np.clip(out, 0, 255)

        elif op_type == "abs_diff":
            base = imgs[0]
            for im in imgs[1:]:
                base = np.abs(base - im)
            out = base

        else:
            messagebox.showerror("Błąd", f"Nieznana operacja: {op_type}")
            return

        out = out.astype(np.uint8)

        self.create_tab_from_array(out, "multi_op", source_tab=tabs[0])
        messagebox.showinfo("Operacje wieloobrazowe", "Operacje wieloobrazowe zakończone pomyślnie.")

    def single_image_number_operation(self, op_type):
        tab = self.require_active_tab()
        if not tab:
            return

        if not self.require_grayscale(tab):
            return

        n = tk.simpledialog.askinteger("Wartość liczbowa", "Podaj liczbę całkowitą:")
        if n is None:
            return

        img = np.array(tab.original, dtype=np.float32)

        # Operacje na obrazach z stałą
        if op_type == "add_no_sat":
            max_val = img.max()
            if max_val + n > 255:
                scale = 255 / (max_val + n)
                img = img * scale
            out = img + n
        elif op_type == "add_sat":
            out = img + n
            out = np.clip(out, 0, 255)
        elif op_type == "mul_no_sat":
            max_val = img.max()
            if max_val * n > 255:
                scale = 255.0 / (max_val * n)
                img = img * scale
            out = img * n
        elif op_type == "mul_sat":
            out = img * n
            out = np.clip(out, 0, 255)
        elif op_type == "div":
            if n == 0:
                messagebox.showerror("Błąd", "Dzielenie przez zero!")
                return
            out = img / n
        else:
            messagebox.showerror("Błąd operacji", f"Nieznany typ operacji: {op_type}")
            return

        out = out.astype(np.uint8)

        self.create_tab_from_array(out, f"{op_type}", source_tab=tab)
        messagebox.showinfo("Operacje z liczbą", "Operacje z liczbą zakończone pomyślnie.")

    def choose_two_images(self):
        tabs = self.choose_images_popup()
        if not tabs or len(tabs) != 2:
            messagebox.showwarning("Błąd", "Musisz wybrać dokładnie dwa obrazy.")
            return None, None
        return tabs[0], tabs[1]

    def logic_not(self):
        tab = self.require_active_tab()
        if not tab:
            return

        img = np.array(tab.original)

        # Binarny
        if set(np.unique(img)).issubset({0, 1}):
            out = 1 - img

        # Binarny 8-bitowy
        elif set(np.unique(img)).issubset({0, 255}):
            out = np.where(img == 0, 255, 0)

        # Monochromatyczny L
        elif tab.original.mode == "L":
            out = 255 - img

        else:
            messagebox.showwarning("Błąd", "Tryb obrazu nieobsługiwany.")
            return

        out = out.astype(np.uint8)
        self.create_tab_from_array(out, "logic_not", source_tab=tab)
        messagebox.showinfo("Operacje logiczne", "Operacja NOT zakończona pomyślnie.")

    def logic_binary(self, op_type):
        A, B = self.choose_two_images()
        if not A:
            return

        imgA = np.array(A.original)
        imgB = np.array(B.original)

        if imgA.shape != imgB.shape:
            messagebox.showwarning("Błąd", "Obrazy muszą mieć ten sam rozmiar.")
            return

        def normalize(img):
            vals = np.unique(img)
            if set(vals).issubset({0, 1}):
                return img.astype(np.uint8)
            if set(vals).issubset({0, 255}):
                return (img // 255).astype(np.uint8)
            return img

        A_norm = normalize(imgA)
        B_norm = normalize(imgB)

        if op_type == "AND":
            out = A_norm & B_norm
        elif op_type == "OR":
            out = A_norm | B_norm
        elif op_type == "XOR":
            out = A_norm ^ B_norm
        else:
            messagebox.showerror("Błąd", "Nieznana operacja logiczna.")
            return

        if set(np.unique(out)).issubset({0, 1}):
            out = out * 255

        self.create_tab_from_array(out, f"logic_{op_type.lower()}", source_tab=A)
        messagebox.showinfo("Operacja logiczna", f"Operacja '{op_type}' zakończona pomyślnie.")

    def to_binary_mask(self):
        tab = self.require_active_tab()
        if not tab:
            return

        img = np.array(tab.original)

        if set(np.unique(img)).issubset({0, 255}):
            messagebox.showinfo("Info", "To już jest obraz binarny.")
            return

        if not self.require_grayscale(tab):
            return

        out = np.where(img >= 128, 255, 0).astype(np.uint8)

        self.create_tab_from_array(out, "to_binary", source_tab=tab)
        messagebox.showinfo("Konwersja maski", "Konwersja maski zakończona pomyślnie..")

    def from_binary_mask(self):
        tab = self.require_active_tab()
        if not tab:
            return

        img = np.array(tab.original)
        unique_vals = set(np.unique(img))

        if unique_vals.issubset({0, 1}):
            out = img * 255
        elif unique_vals.issubset({0, 255}):
            out = img.astype(np.uint8)
        else:
            messagebox.showwarning("Błąd", "Obraz nie jest binarny.")
            return

        self.create_tab_from_array(out, "to_8bit", source_tab=tab)
        messagebox.showinfo("Konwersja maski", "Konwersja maski zakończona pomyślnie.")

    def filter_select(self, filter_type):
        tab = self.require_active_tab()
        if not tab:
            return

        if not self.require_grayscale(tab):
            return

        win = tk.Toplevel(self.root)
        win.title("Parametry filtra")
        win.geometry("250x300")

        # Wybór maski
        tk.Label(win, text="Wybierz maskę:").pack()

        mask_var = tk.StringVar(win)

        options = None
        preview_masks = {}

        if filter_type == "mean":
            options = {
                "Średnia 3×3": "mean3",
                "Średnia 5×5": "mean5"
            }

        elif filter_type == "weighted":
            options = {
                "Wagowa 3×3": "weight3"
            }
            preview_masks = {
                "Wagowa 3×3": "1  2  1\n2  4  2\n1  2  1\n(× 1/16)"
            }

        elif filter_type == "gauss":
            options = {
                "Gauss 3×3": "gauss3"
            }
            preview_masks = {
                "Gauss 3×3": "1  2  1\n2  4  2\n1  2  1"
            }

        elif filter_type == "laplace":
            options = {
                "Laplace 1": "laplace1",
                "Laplace 2": "laplace2",
                "Laplace 3": "laplace3"
            }
            preview_masks = {
                "Laplace 1": " 0  -1   0\n-1   4  -1\n 0  -1   0",
                "Laplace 2": "-1  -1  -1\n-1   8  -1\n-1  -1  -1",
                "Laplace 3": " 1  -2   1\n-2   4  -2\n 1  -2   1"
            }

        elif filter_type == "prewitt":
            options = {
                "N": "prewitt_n",
                "NE": "prewitt_ne",
                "E": "prewitt_e",
                "SE": "prewitt_se",
                "S": "prewitt_s",
                "SW": "prewitt_sw",
                "W": "prewitt_w",
                "NW": "prewitt_nw"
            }
            preview_masks = {
                "N" : "-1 -1 -1\n 0  0  0\n 1  1  1",
                "NE": " 0 -1 -1\n 1  0 -1\n 1  1  0",
                "E" : "-1  0  1\n-1  0  1\n-1  0  1",
                "SE": "-1 -1  0\n-1  0  1\n 0  1  1",
                "S" : " 1  1  1\n 0  0  0\n-1 -1 -1",
                "SW": " 0  1  1\n-1  0  1\n-1 -1  0",
                "W" : " 1  0 -1\n 1  0 -1\n 1  0 -1",
                "NW": " 1  1  0\n 1  0 -1\n 0 -1 -1"
            }

        elif filter_type == "sobel":
            options = {
                "Sobel X": "sobel_x",
                "Sobel Y": "sobel_y"
            }
            preview_masks = {
                "Sobel X": "-1  0  1\n-2  0  2\n-1  0  1",
                "Sobel Y": "-1 -2 -1\n 0  0  0\n 1  2  1"
            }

        mask_var.set(list(options.keys())[0])
        mask_menu = tk.OptionMenu(win, mask_var, *options.keys())
        mask_menu.pack()

        if preview_masks:
            mask_preview = tk.Label(
                win,
                text=preview_masks[mask_var.get()],
                font=("Courier New", 10),
                justify="left"
            )
            mask_preview.pack(pady=10)

            def on_mask_change(*args):
                mask_preview.config(text=preview_masks[mask_var.get()])

            mask_var.trace_add("write", on_mask_change)

        tk.Label(win, text="Wybierz tryb brzegów:").pack()

        border_var = tk.StringVar(win)
        border_var.set("BORDER_CONSTANT")

        border_menu = tk.OptionMenu(
            win,
            border_var,
            "BORDER_CONSTANT",
            "OUTPUT_CONSTANT",
            "BORDER_REFLECT"
        )
        border_menu.pack()

        const_label = tk.Label(win, text="Wartość stała (jeśli dotyczy):")
        const_entry = tk.Entry(win)

        const_label.pack()
        const_entry.pack()

        def run_filter():
            mask_key = options[mask_var.get()]
            border_mode = border_var.get()

            try:
                const_val = int(const_entry.get()) if const_entry.get() else 0
            except:
                const_val = 0

            win.destroy()
            self.apply_filter(tab, filter_type, mask_key, border_mode, const_val)

        tk.Button(win, text="OK", command=run_filter).pack(pady=10)

    def apply_filter(self, tab, filter_type, mask_key, border_mode, const_val):
        img = np.array(tab.original).astype(np.uint8)

        mask = mask_array.get_mask(mask_key)

        if border_mode == "BORDER_CONSTANT":
            border = cv2.BORDER_CONSTANT
        elif border_mode == "BORDER_REFLECT":
            border = cv2.BORDER_REFLECT
        else:
            # OUTPUT_CONSTANT
            border = cv2.BORDER_REFLECT

        out = cv2.filter2D(
            img.astype(np.float32),
            ddepth=cv2.CV_32F,
            kernel=mask,
            borderType=border
        )

        out = np.clip(out, 0, 255).astype(np.uint8)

        if border_mode == "OUTPUT_CONSTANT":
            k = mask.shape[0] // 2
            out[:k, :] = const_val
            out[-k:, :] = const_val
            out[:, :k] = const_val
            out[:, -k:] = const_val

        self.create_tab_from_array(out, f"{filter_type}", source_tab=tab)
        messagebox.showinfo("Filtr", f"Filtrowanie {filter_type} zakończona pomyślnie.")

    def median_filter_select(self):
        tab = self.require_active_tab()
        if not tab:
            return

        if not self.require_grayscale(tab):
            return

        win = tk.Toplevel(self.root)
        win.title("Parametry filtra medianowego")
        win.geometry("250x300")

        tk.Label(win, text="Wybierz rozmiar maski:").pack()

        size_var = tk.StringVar(win)
        size_var.set("3x3")

        size_menu = tk.OptionMenu(win, size_var, "3x3", "5x5", "7x7", "9x9")
        size_menu.pack()

        tk.Label(win, text="Wybierz tryb brzegów:").pack()

        border_var = tk.StringVar(win)
        border_var.set("BORDER_REFLECT")

        border_menu = tk.OptionMenu(
            win,
            border_var,
            "BORDER_CONSTANT",
            "OUTPUT_CONSTANT",
            "BORDER_REFLECT"
        )
        border_menu.pack()

        tk.Label(win, text="Stała n (jeśli dotyczy):").pack()
        const_entry = tk.Entry(win)
        const_entry.pack()

        def apply():
            size_str = size_var.get()
            ksize = int(size_str[0])
            mode = border_var.get()
            try:
                const_val = int(const_entry.get()) if const_entry.get() else 0
            except:
                const_val = 0
            win.destroy()
            self.apply_median(tab, ksize, mode, const_val)

        tk.Button(win, text="OK", command=apply).pack(pady=10)

    def apply_median(self, tab, ksize, border_mode, const_val):
        img = np.array(tab.original).astype(np.uint8)

        pad = ksize // 2

        if border_mode == "BORDER_CONSTANT":
            padded = cv2.copyMakeBorder(
                img, pad, pad, pad, pad,
                cv2.BORDER_CONSTANT, value=const_val
            )

        elif border_mode == "BORDER_REFLECT":
            padded = cv2.copyMakeBorder(
                img, pad, pad, pad, pad,
                cv2.BORDER_REFLECT
            )

        else:
            # OUTPUT_CONSTANT
            padded = cv2.copyMakeBorder(
                img, pad, pad, pad, pad,
                cv2.BORDER_REFLECT
            )

        out = np.zeros_like(img)
        H, W = img.shape
        for y in range(H):
            for x in range(W):
                window = padded[y:y + ksize, x:x + ksize]
                out[y, x] = np.median(window)
        out = out.astype(np.uint8)

        # OUTPUT_CONSTANT
        if border_mode == "OUTPUT_CONSTANT":
            out[:pad, :] = const_val
            out[-pad:, :] = const_val
            out[:, :pad] = const_val
            out[:, -pad:] = const_val

        self.create_tab_from_array(out, "median", source_tab=tab)
        messagebox.showinfo("Filtr medianowy", "Filtrowanie medianowe zakończona pomyślnie.")

    def canny_select(self):
        tab = self.require_active_tab()
        if not tab:
            return

        if not self.require_grayscale(tab):
            return

        win = tk.Toplevel(self.root)
        win.title("Parametry operatora Canny")
        win.geometry("250x300")

        tk.Label(win, text="Dolny próg (threshold1):").pack()
        t1_entry = tk.Entry(win)
        t1_entry.insert(0, "50")
        t1_entry.pack()

        tk.Label(win, text="Górny próg (threshold2):").pack()
        t2_entry = tk.Entry(win)
        t2_entry.insert(0, "150")
        t2_entry.pack()

        tk.Label(win, text="Apertura (3, 5 lub 7):").pack()
        ap_entry = tk.Entry(win)
        ap_entry.insert(0, "3")
        ap_entry.pack()

        def apply():
            try:
                t1 = int(t1_entry.get())
                t2 = int(t2_entry.get())
                ap = int(ap_entry.get())
                if ap not in (3, 5, 7):
                    raise ValueError
            except:
                messagebox.showerror("Błąd", "Niepoprawne wartości parametrów.")
                return

            win.destroy()
            self.apply_canny(tab, t1, t2, ap)

        tk.Button(win, text="OK", command=apply).pack(pady=10)

    def apply_canny(self, tab, t1, t2, aperture):
        img = np.array(tab.original).astype(np.uint8)

        edges = cv2.Canny(img, t1, t2, apertureSize=aperture)

        self.create_tab_from_array(edges, "canny", source_tab=tab)

        messagebox.showinfo("Canny","Detekcja krawędzi zakończona pomyślnie.")

    # LABORATORIUM 3
    def stretch_popup(self):
        tab = self.require_active_tab()
        if not tab:
            return

        if not self.require_grayscale(tab):
            return

        win = tk.Toplevel(self.root)
        win.title("Rozciąganie histogramu")
        win.geometry("300x250")

        labels = ["p1 (wejściowe min)", "p2 (wejściowe max)", "q3 (wyjściowe min)", "q4 (wyjściowe max)"]
        entries = []

        for text in labels:
            tk.Label(win, text=text).pack()
            e = tk.Entry(win)
            e.pack()
            entries.append(e)

        def apply():
            try:
                p1 = int(entries[0].get())
                p2 = int(entries[1].get())
                q3 = int(entries[2].get())
                q4 = int(entries[3].get())
            except:
                messagebox.showerror("Błąd", "Wprowadź poprawne wartości liczbowe.")
                return

            if p1 >= p2 or q3 >= q4:
                messagebox.showerror("Błąd", "Zakresy muszą spełniać p1 < p2 oraz q3 < q4.")
                return

            win.destroy()
            self.apply_histogram_stretch(tab, p1, p2, q3, q4)

        tk.Button(win, text="OK", command=apply).pack(pady=10)

    def apply_histogram_stretch(self, tab, p1, p2, q3, q4):
        img = np.array(tab.original).astype(np.float32)

        out = np.zeros_like(img)

        mask1 = img < p1
        mask2 = (img >= p1) & (img <= p2)
        mask3 = img > p2

        out[mask1] = q3
        out[mask3] = q4
        out[mask2] = ((img[mask2] - p1) * (q4 - q3) / (p2 - p1)) + q3

        self.create_tab_from_array(out, "hist_stretch", source_tab=tab)
        messagebox.showinfo("Rozciąganie histogramu", "Rozciąganie histogramu zakończone pomyślnie.")

    def threshold_two_popup(self):
        tab = self.require_active_tab()
        if not tab:
            return

        if not self.require_grayscale(tab):
            return

        win = tk.Toplevel(self.root)
        win.title("Progowanie z dwoma progami")
        win.geometry("300x200")

        tk.Label(win, text="Próg 1 (t1):").pack()
        t1_entry = tk.Entry(win)
        t1_entry.pack()

        tk.Label(win, text="Próg 2 (t2):").pack()
        t2_entry = tk.Entry(win)
        t2_entry.pack()

        def apply():
            try:
                t1 = int(t1_entry.get())
                t2 = int(t2_entry.get())
            except:
                messagebox.showerror("Błąd", "Podaj poprawne wartości liczbowe.")
                return

            if t1 >= t2:
                messagebox.showerror("Błąd", "Musi być t1 < t2.")
                return

            win.destroy()
            self.apply_threshold_two(tab, t1, t2)

        tk.Button(win, text="OK", command=apply).pack(pady=10)

    def apply_threshold_two(self, tab, t1, t2):
        img = np.array(tab.original)

        out = np.zeros_like(img)

        out[(img >= t1) & (img < t2)] = 255
        out[img >= t2] = 0

        self.create_tab_from_array(out, "threshold_two", source_tab=tab)
        messagebox.showinfo("Progowanie", "Progowanie dwuetapowe zakończone pomyślnie.")

    def threshold_otsu(self):
        tab = self.require_active_tab()
        if not tab:
            return

        if not self.require_grayscale(tab):
            return

        img = np.array(tab.original).astype(np.uint8)

        thresh, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self.create_tab_from_array(out, "threshold_otsu", source_tab=tab)
        messagebox.showinfo("Otsu", f"Progowanie Otsu zakończone pomyślnie. Wyznaczony próg: {thresh}")

    def threshold_adaptive_popup(self):
        tab = self.require_active_tab()
        if not tab:
            return

        if not self.require_grayscale(tab):
            return

        win = tk.Toplevel(self.root)
        win.title("Adaptive threshold")
        win.geometry("300x220")

        tk.Label(win, text="Rozmiar bloku (nieparzysty, np. 5):").pack()
        block_entry = tk.Entry(win)
        block_entry.insert(0, "5")
        block_entry.pack()

        tk.Label(win, text="C (wartość odejmowana, np. 5):").pack()
        c_entry = tk.Entry(win)
        c_entry.insert(0, "5")
        c_entry.pack()

        def apply():
            try:
                block = int(block_entry.get())
                C = int(c_entry.get())
            except:
                messagebox.showerror("Błąd", "Podaj liczby całkowite.")
                return

            if block % 2 == 0 or block < 3:
                messagebox.showerror("Błąd", "Rozmiar bloku musi być nieparzysty i ≥ 3.")
                return

            win.destroy()
            self.apply_threshold_adaptive(tab, block, C)

        tk.Button(win, text="OK", command=apply).pack(pady=10)

    def apply_threshold_adaptive(self, tab, block, C):
        img = np.array(tab.original).astype(np.uint8)

        out = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block,
            C
        )

        self.create_tab_from_array(out, "threshold_adaptive", source_tab=tab)
        messagebox.showinfo("Adaptive threshold", "Progowanie adaptacyjne zakończone pomyślnie.")

    def morph_select(self, operation):
        tab = self.require_active_tab()
        if not tab:
            return

        if not self.require_grayscale(tab):
            return

        win = tk.Toplevel(self.root)
        win.title("Parametry morfologii")
        win.geometry("300x200")

        tk.Label(win, text="Element strukturalny").pack()

        elem_var = tk.StringVar(win)
        elem_var.set("Kwadrat 3×3")

        elem_menu = tk.OptionMenu(win, elem_var, "Kwadrat 3×3", "Krzyż 3×3")
        elem_menu.pack()

        def apply():
            element = elem_var.get()
            win.destroy()
            self.apply_morph(tab, operation, element)

        tk.Button(win, text="OK", command=apply).pack(pady=10)

    def apply_morph(self, tab, operation, element):
        img = np.array(tab.original).astype(np.uint8)

        if element == "Kwadrat 3×3":
            kernel = np.ones((3, 3), np.uint8)
        else:
            kernel = np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ], dtype=np.uint8)

        # Operacje morfologiczne
        if operation == "erode":
            out = cv2.erode(img, kernel, iterations=1)
        elif operation == "dilate":
            out = cv2.dilate(img, kernel, iterations=1)
        elif operation == "open":
            out = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            out = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        else:
            messagebox.showwarning("Błąd", "Nieznana operacja morfologiczna.")
            return

        self.create_tab_from_array(out, f"{operation}", source_tab=tab)
        messagebox.showinfo("Morfologia", f"Operacja '{operation}' zakończona pomyślnie.")

    def skeletonize(self):
        tab = self.require_active_tab()
        if not tab:
            return

        img = np.array(tab.original)

        unique_vals = set(np.unique(img))
        if not unique_vals.issubset({0, 255}):
            messagebox.showwarning("Błąd", "Szkieletyzacja działa tylko na obrazach binarnych.")
            return

        # Konwersja do 0 - 1 dla obliczeń
        img = (img // 255).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skeleton = np.zeros_like(img)
        done = False

        while not done:
            eroded = cv2.erode(img, kernel)
            opened = cv2.dilate(eroded, kernel)

            temp = img - opened
            skeleton = cv2.bitwise_or(skeleton, temp)

            img = eroded.copy()

            if cv2.countNonZero(img) == 0:
                done = True

        # Powrót do 0 - 255
        out = (skeleton * 255).astype(np.uint8)

        self.create_tab_from_array(out, "skeletonize", source_tab=tab)
        messagebox.showinfo("Szkieletyzacja", "Szkieletyzacja zakończona pomyślnie.")

    # LABORATORIUM 4
    def save_features_to_file(self, features):
        path = filedialog.asksaveasfilename(
            title="Zapisz cechy obiektu",
            defaultextension=".csv",
            filetypes=[("Plik CSV", "*.csv"), ("Plik TXT", "*.txt")]
        )

        if not path:
            return False

        with open(path, "w", encoding="utf-8") as f:
            f.write("feature;value\n")
            for key in sorted(features.keys()):
                f.write(f"{key};{features[key]}\n")

        return True

    def compute_binary_features(self, cnt):
        features = {}
        M = cv2.moments(cnt)

        for k, v in M.items():
            features[f"moment_{k}"] = v

        # Środek masy
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = 0, 0
        features["centroid_x"] = cx
        features["centroid_y"] = cy

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        features["area"] = area
        features["perimeter"] = perimeter

        # Współczynniki kształtu
        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = w / h if h != 0 else 0

        extent = area / (w * h) if w * h != 0 else 0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0

        equivalent_diameter = np.sqrt(4 * area / np.pi) if area != 0 else 0

        features["aspectRatio"] = aspect_ratio
        features["extent"] = extent
        features["solidity"] = solidity
        features["equivalentDiameter"] = equivalent_diameter

        return features

    def analyze_binary_object(self):
        tab = self.require_active_tab()
        if not tab:
            return

        img = np.array(tab.original)

        unique_vals = set(np.unique(img))
        if not unique_vals.issubset({0, 255}):
            messagebox.showwarning("Błąd", "Analiza działa tylko na obrazach binarnych (0 i 255).")
            return

        # Konwersja do 0/1
        bin_img = (img // 255).astype(np.uint8)

        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            messagebox.showwarning("Błąd", "Nie znaleziono obiektu na obrazie.")
            return

        cnt = max(contours, key=cv2.contourArea)

        features = self.compute_binary_features(cnt)

        saved = self.save_features_to_file(features)
        if saved:
            messagebox.showinfo("Analiza zakończona", "Analiza obrazu zakończona pomyślnie. Wynik zapisano do pliku.")

    def apply_hough_lines(self, tab, threshold, min_len, max_gap):
        img = np.array(tab.original).astype(np.uint8)

        edges = cv2.Canny(img, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_len,
            maxLineGap=max_gap
        )

        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

        self.create_tab_from_array(out, "hough_lines", source_tab=tab)
        messagebox.showinfo("Hough", "Detekcja linii transformatą Hougha zakończona pomyślnie.")

    def hough_lines_popup(self):
        tab = self.require_active_tab()
        if not tab:
            return

        if not self.require_grayscale(tab):
            return

        win = tk.Toplevel(self.root)
        win.title("Transformata Hougha – parametry")
        win.geometry("300x260")

        tk.Label(win, text="threshold (min. głosów):").pack()
        th_entry = tk.Entry(win)
        th_entry.insert(0, "100")
        th_entry.pack()

        tk.Label(win, text="minLineLength:").pack()
        minlen_entry = tk.Entry(win)
        minlen_entry.insert(0, "50")
        minlen_entry.pack()

        tk.Label(win, text="maxLineGap:").pack()
        gap_entry = tk.Entry(win)
        gap_entry.insert(0, "10")
        gap_entry.pack()

        def apply():
            try:
                threshold = int(th_entry.get())
                min_len = int(minlen_entry.get())
                max_gap = int(gap_entry.get())
            except:
                messagebox.showerror("Błąd", "Podaj poprawne wartości liczbowe.")
                return

            win.destroy()
            self.apply_hough_lines(tab, threshold, min_len, max_gap)

        tk.Button(win, text="OK", command=apply).pack(pady=15)

    # MINI-PROJEKT
    def advanced_histogram_transform(self):
        tab = self.require_active_tab()
        if not tab:
            return

        if not self.require_grayscale(tab):
            return

        win = tk.Toplevel(self.root)
        win.title("Transformacja histogramu")
        win.geometry("350x400")

        mode_var = tk.StringVar(value="linear")

        radio_frame = tk.Frame(win)
        radio_frame.pack(pady=5)

        tk.Label(radio_frame, text="Wybierz typ transformacji:").pack()

        tk.Radiobutton(radio_frame, text="Liniowa",
                       variable=mode_var, value="linear",
                       command=lambda: update_fields()).pack()

        tk.Radiobutton(radio_frame, text="Gamma",
                       variable=mode_var, value="gamma",
                       command=lambda: update_fields()).pack()

        params_frame = tk.Frame(win)
        params_frame.pack(pady=10)

        linear_frame = tk.Frame(params_frame)
        gamma_frame = tk.Frame(params_frame)

        tk.Label(linear_frame, text="p1:").pack()
        p1_entry = tk.Entry(linear_frame)
        p1_entry.pack()
        tk.Label(linear_frame, text="p2:").pack()
        p2_entry = tk.Entry(linear_frame)
        p2_entry.pack()
        tk.Label(linear_frame, text="q3:").pack()
        q3_entry = tk.Entry(linear_frame)
        q3_entry.pack()
        tk.Label(linear_frame, text="q4:").pack()
        q4_entry = tk.Entry(linear_frame)
        q4_entry.pack()

        tk.Label(gamma_frame, text="Gamma:").pack()
        gamma_entry = tk.Entry(gamma_frame)
        gamma_entry.insert(0, "1.0")
        gamma_entry.pack()

        def update_fields():
            for widget in params_frame.winfo_children():
                widget.pack_forget()

            if mode_var.get() == "linear":
                linear_frame.pack()
            else:
                gamma_frame.pack()

        update_fields()

        button_frame = tk.Frame(win)
        button_frame.pack(pady=10)

        def apply():
            img = np.array(tab.original).astype(np.float32)

            if mode_var.get() == "linear":
                try:
                    p1 = float(p1_entry.get())
                    p2 = float(p2_entry.get())
                    q3 = float(q3_entry.get())
                    q4 = float(q4_entry.get())
                except:
                    messagebox.showerror("Błąd", "Niepoprawne parametry.")
                    return

                if not (0 <= p1 < p2 <= 255):
                    messagebox.showerror("Błąd", "Zakres wejściowy musi spełniać 0 ≤ p1 < p2 ≤ 255.")
                    return

                if not (0 <= q3 < q4 <= 255):
                    messagebox.showerror("Błąd", "Zakres wyjściowy musi spełniać 0 ≤ q3 < q4 ≤ 255.")
                    return

                out = np.zeros_like(img)

                mask1 = img < p1
                mask2 = (img >= p1) & (img <= p2)
                mask3 = img > p2

                out[mask1] = q3
                out[mask2] = ((img[mask2] - p1) * (q4 - q3) / (p2 - p1)) + q3
                out[mask3] = q4

            else:
                try:
                    gamma = float(gamma_entry.get())
                    if gamma <= 0:
                        raise ValueError
                except:
                    messagebox.showerror("Błąd", "Gamma musi być dodatnia.")
                    return

                img_norm = img / 255.0
                out = 255 * np.power(img_norm, 1.0 / gamma)

            out = np.clip(out, 0, 255).astype(np.uint8)

            if mode_var.get() == "linear":
                op_name = "linear_transform"
            else:
                op_name = "gamma_transform"

            self.create_tab_from_array(out, op_name, source_tab=tab)
            messagebox.showinfo("Transformacja", "Transformacja histogramu zakończona pomyślnie.")
            win.destroy()

        tk.Button(button_frame, text="Zastosuj", command=apply).pack()

    def show_about(self):
        info_text = (
            "Aplikacja do przetwarzania obrazów cyfrowych\n\n"
            "Projekt wykonany w ramach przedmiotu:\n"
            "Algorytmy Przetwarzania Obrazów\n\n"
            "Autor: Filip Stefański\n"
            "Numer albumu: 21075\n"
            "Rok akademicki: 2025/2026"
        )
        messagebox.showinfo("O autorze i aplikacji", info_text)

# Uruchomienie aplikacji
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = ImageApp(root)
    root.mainloop()
