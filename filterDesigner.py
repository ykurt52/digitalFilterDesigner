import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import tkinter as tk
import numpy as np
from scipy import signal
from datetime import datetime
from tkinter import filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

FIR_MODELS = ["firwin"]
IIR_MODELS = ["bessel", "butter", "cheby1", "cheby2", "ellip"]
FIR_WINDOWS = ["hamming", "hann", "blackman", "bartlett", "kaiser", "boxcar"]
IIR_STRUCTS = ["Direct Form I", "Direct Form II", "SOS (biquad)"]

def quantize_coefficients(coeffs, quant_type='float'):
    if quant_type == 'float':
        return coeffs.astype(np.float32)
    elif quant_type == 'double':
        return coeffs.astype(np.float64)
    elif quant_type == 'int16':
        return np.round(coeffs * 32767).astype(np.int16)
    elif quant_type == 'int32':
        return np.round(coeffs * 2147483647).astype(np.int32)
    else:
        raise ValueError("Unsupported quantization type!")

def coefficients_to_c_array(coeffs, name="FilterCoef", datatype="float", precision=17):
    fmt = f"%.{precision}g" if 'float' in datatype or 'double' in datatype else "%d"
    elements = (fmt % v for v in coeffs)
    arr = ",\n    ".join(elements)
    cstr = f"{datatype} {name}[{len(coeffs)}] = {{\n    {arr}\n}};\n"
    return cstr

def moving_average(x, w):
    w = int(w)
    if w < 1: return x
    return np.convolve(x, np.ones(w)/w, mode='same')

def median_filter(x, w):
    w = int(w)
    if w < 1: return x
    from scipy.ndimage import median_filter
    return median_filter(x, size=w, mode='nearest')

class FilterDesignerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Digital Filter Designer")
        root.geometry("1320x820")
        style = ttkb.Style(theme="flatly")

        self.main = ttkb.Frame(root, padding=(10,12,12,10))
        self.main.pack(fill=BOTH, expand=YES)

        self.left = ttkb.Frame(self.main)
        self.left.pack(side=LEFT, fill=Y, padx=(8,12), pady=8, expand=False)
        self.right = ttkb.Frame(self.main)
        self.right.pack(side=LEFT, fill=BOTH, expand=YES, padx=(8,8), pady=8)

        ttkb.Label(self.left, text="Digital Filter Designer", font=("Segoe UI", 20, "bold")).pack(pady=(2,10))

        # --------- Filter Parameters -------------
        param_frame = ttkb.LabelFrame(self.left, text="Filter Parameters", bootstyle=INFO, padding=(14,7,14,7))
        param_frame.pack(fill=X, pady=(0,8))

        param_frame.columnconfigure(0, weight=0, minsize=145)
        param_frame.columnconfigure(1, weight=1, minsize=210)
        param_frame.columnconfigure(2, weight=0, minsize=135)

        row = 0
        ttkb.Label(param_frame, text="Filter Family:", font=("Segoe UI", 11)).grid(row=row, column=0, sticky=E, pady=3, padx=5)
        self.filter_family = ttkb.Combobox(param_frame, values=["FIR", "IIR"], state="readonly", font=("Segoe UI", 11), width=10)
        self.filter_family.grid(row=row, column=1, sticky=W+E, padx=3, pady=(0,10))
        self.filter_family.current(0)
        self.filter_family.bind("<<ComboboxSelected>>", self.update_model_list)
        row += 1
        ttkb.Label(param_frame, text="Filter Type:", font=("Segoe UI", 11)).grid(row=row, column=0, sticky=E, pady=3, padx=5)
        self.filter_type = ttkb.Combobox(param_frame, values=["lowpass", "highpass", "bandpass", "bandstop"], state="readonly", font=("Segoe UI", 11), width=10)
        self.filter_type.current(0)
        self.filter_type.grid(row=row, column=1, sticky=W+E, padx=3, pady=(0,10))
        row += 1
        ttkb.Label(param_frame, text="Filter Model:", font=("Segoe UI", 11)).grid(row=row, column=0, sticky=E, pady=3, padx=5)
        self.filter_model = ttkb.Combobox(param_frame, values=FIR_MODELS, state="readonly", font=("Segoe UI", 11), width=12)
        self.filter_model.current(0)
        self.filter_model.grid(row=row, column=1, sticky=W+E, padx=3, pady=(0,10))
        self.filter_model.bind("<<ComboboxSelected>>", self.update_extra_fields)
        row += 1
        self.firwin_label = ttkb.Label(param_frame, text="FIR Window:", font=("Segoe UI", 11))
        self.firwin = ttkb.Combobox(param_frame, values=FIR_WINDOWS, state="readonly", font=("Segoe UI", 11), width=10)
        self.firwin.grid(row=row, column=1, sticky=W+E, padx=3, pady=(0,10))
        self.firwin.current(0)
        row += 1
        self.iir_struct_label = ttkb.Label(param_frame, text="IIR Structure:", font=("Segoe UI", 11))
        self.iir_struct = ttkb.Combobox(param_frame, values=IIR_STRUCTS, state="readonly", font=("Segoe UI", 11), width=14)
        self.iir_struct.grid(row=row, column=1, sticky=W+E, padx=3, pady=(0,10))
        self.iir_struct.current(0)
        row += 1
        ttkb.Label(param_frame, text="Order:", font=("Segoe UI", 11)).grid(row=row, column=0, sticky=E, pady=3, padx=5)
        self.order = ttkb.Entry(param_frame, font=("Segoe UI", 11), width=7)
        self.order.insert(0, "8")
        self.order.grid(row=row, column=1, sticky=W+E, padx=3, pady=(0,10))
        row += 1
        ttkb.Label(param_frame, text="Sampling Freq (Hz):", font=("Segoe UI", 11)).grid(row=row, column=0, sticky=E, pady=3, padx=5)
        self.fs = ttkb.Entry(param_frame, font=("Segoe UI", 11), width=9)
        self.fs.insert(0, "1000")
        self.fs.grid(row=row, column=1, sticky=W+E, padx=3, pady=(0,10))
        row += 1
        ttkb.Label(param_frame, text="Cutoff Freq (Hz):", font=("Segoe UI", 11)).grid(row=row, column=0, sticky=E, pady=3, padx=5)
        self.cutoff = ttkb.Entry(param_frame, font=("Segoe UI", 11), width=14)
        self.cutoff.insert(0, "50")
        self.cutoff.grid(row=row, column=1, sticky=W+E, padx=3, pady=(0,10))
        ttkb.Label(param_frame, text="(Band: Use 40,60)", font=("Segoe UI", 9, "italic")).grid(row=row, column=2, sticky=W)
        row += 1
        ttkb.Label(param_frame, text="Quantization:", font=("Segoe UI", 11)).grid(row=row, column=0, sticky=E, pady=3, padx=5)
        self.quant = ttkb.Combobox(param_frame, values=["float", "double", "int16", "int32"], state="readonly", font=("Segoe UI", 11), width=10)
        self.quant.current(0)
        self.quant.grid(row=row, column=1, sticky=W+E, padx=3, pady=(0,10))
        row += 1
        ttkb.Label(param_frame, text="Filter Name (C id):", font=("Segoe UI", 11)).grid(row=row, column=0, sticky=E, pady=3, padx=5)
        self.filter_name = ttkb.Entry(param_frame, font=("Segoe UI", 11), width=16)
        self.filter_name.insert(0, "myFilter")
        self.filter_name.grid(row=row, column=1, sticky=W+E, padx=3, pady=(0,10))
        row += 1
        ttkb.Label(param_frame, text="User Header/Comment:", font=("Segoe UI", 11)).grid(row=row, column=0, sticky=E, pady=3, padx=5)
        self.user_header = ttkb.Entry(param_frame, font=("Segoe UI", 11), width=34)
        self.user_header.insert(0, "Your company, author, notes...")
        self.user_header.grid(row=row, column=1, columnspan=2, sticky=W+E, padx=3, pady=(0,10))
        row += 1
        self.rp_label = ttkb.Label(param_frame, text="Passband Ripple (dB):", font=("Segoe UI", 11))
        self.rp = ttkb.Entry(param_frame, font=("Segoe UI", 11), width=8)
        self.rs_label = ttkb.Label(param_frame, text="Stopband Atten (dB):", font=("Segoe UI", 11))
        self.rs = ttkb.Entry(param_frame, font=("Segoe UI", 11), width=8)
        self.rp.insert(0, "1")
        self.rs.insert(0, "40")

        # --------- Moving Average / Median Filter Checkbox'ları -------------
        filter_opts = ttkb.LabelFrame(self.left, text="Show extra filters on example plot", bootstyle=WARNING, padding=(12,8,8,7))
        filter_opts.pack(fill=X, pady=(0,7))
        self.show_ma = ttkb.BooleanVar(value=False)
        self.show_med = ttkb.BooleanVar(value=False)
        self.ma_chk = ttkb.Checkbutton(filter_opts, text="Moving Average", variable=self.show_ma, bootstyle=SUCCESS)
        self.ma_chk.grid(row=0, column=0, sticky=W, padx=2)
        self.med_chk = ttkb.Checkbutton(filter_opts, text="Median Filter", variable=self.show_med, bootstyle=PRIMARY)
        self.med_chk.grid(row=1, column=0, sticky=W, padx=2)

        ttkb.Label(filter_opts, text="Window size:", font=("Segoe UI", 10)).grid(row=0, column=1, sticky=E, padx=4)
        self.ma_size = ttkb.Entry(filter_opts, font=("Segoe UI", 10), width=5)
        self.ma_size.insert(0, "9")
        self.ma_size.grid(row=0, column=2, sticky=W, padx=2)

        ttkb.Label(filter_opts, text="Window size:", font=("Segoe UI", 10)).grid(row=1, column=1, sticky=E, padx=4)
        self.med_size = ttkb.Entry(filter_opts, font=("Segoe UI", 10), width=5)
        self.med_size.insert(0, "9")
        self.med_size.grid(row=1, column=2, sticky=W, padx=2)

        # --------- Buttons and C code output -------------
        action_frame = ttkb.Frame(self.left)
        action_frame.pack(fill=X, pady=(12,6))
        gen_btn = ttkb.Button(action_frame, text="Generate & Plot", bootstyle=SUCCESS, width=19, command=self.generate)
        gen_btn.pack(side=LEFT, padx=3)
        export_c_btn = ttkb.Button(action_frame, text="Export .c/.h File", bootstyle=INFO, width=17, command=self.export_c_file)
        export_c_btn.pack(side=LEFT, padx=3)
        copy_btn = ttkb.Button(action_frame, text="Copy C Code", bootstyle=SECONDARY, width=14, command=self.copy_c_code)
        copy_btn.pack(side=LEFT, padx=3)
        example_btn = ttkb.Button(action_frame, text="Show Example Signal Filtering", bootstyle=PRIMARY, width=31, command=self.show_example_signal)
        example_btn.pack(side=LEFT, padx=3)

        ccard = ttkb.Labelframe(self.left, text="Generated C Code", bootstyle=PRIMARY)
        ccard.pack(fill=BOTH, pady=(4,6), padx=2, expand=True)
        self.result = tk.Text(ccard, height=19, width=66, font=("Consolas", 10), bg="#faf9fa")
        self.result.pack(fill=BOTH, expand=YES, padx=7, pady=5)

        # --- RIGHT: Plots ---
        self.fig = Figure(figsize=(8.5, 3.1), dpi=98)
        self.ax_mag = self.fig.add_subplot(211)
        self.ax_phase = self.fig.add_subplot(212)
        self.fig.subplots_adjust(hspace=0.42)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=YES, pady=3)

        self.fig_imp = Figure(figsize=(8.5, 1.7), dpi=98)
        self.ax_imp = self.fig_imp.add_subplot(111)
        self.canvas_imp = FigureCanvasTkAgg(self.fig_imp, master=self.right)
        self.canvas_imp.get_tk_widget().pack(fill=BOTH, expand=NO, pady=(2,10))

        self.example_fig = None
        self.example_canvas = None

        self.update_model_list()
        self.update_extra_fields()

    def update_model_list(self, *args):
        family = self.filter_family.get()
        if family == "FIR":
            self.filter_model['values'] = FIR_MODELS
            self.filter_model.current(0)
            self.firwin_label.grid(row=4, column=0, sticky=E)
            self.firwin.grid(row=4, column=1, sticky=W+E)
            self.iir_struct_label.grid_remove()
            self.iir_struct.grid_remove()
        else:
            self.filter_model['values'] = IIR_MODELS
            self.filter_model.current(0)
            self.firwin_label.grid_remove()
            self.firwin.grid_remove()
            self.iir_struct_label.grid(row=4, column=0, sticky=E)
            self.iir_struct.grid(row=4, column=1, sticky=W+E)
        self.update_extra_fields()

    def update_extra_fields(self, *args):
        m = self.filter_model.get()
        self.rp_label.grid_remove()
        self.rp.grid_remove()
        self.rs_label.grid_remove()
        self.rs.grid_remove()
        if m in ["cheby1", "ellip"]:
            self.rp_label.grid(row=17, column=0, sticky=E)
            self.rp.grid(row=17, column=1, sticky=W+E)
        if m in ["cheby2", "ellip"]:
            self.rs_label.grid(row=18, column=0, sticky=E)
            self.rs.grid(row=18, column=1, sticky=W+E)

    def generate(self):
        try:
            ftype = self.filter_type.get()
            model = self.filter_model.get()
            family = self.filter_family.get()
            order = int(self.order.get())
            fs = float(self.fs.get())
            cutoff_raw = self.cutoff.get()
            if ftype in ["bandpass", "bandstop"]:
                cutoff = [float(x.strip()) for x in cutoff_raw.split(",")]
                cutoff_str = ",".join([str(c) for c in cutoff])
            else:
                cutoff = float(cutoff_raw)
                cutoff_str = str(cutoff)
            quant = self.quant.get()
            fname = self.filter_name.get().strip()
            user_header = self.user_header.get().strip()
            rp = float(self.rp.get()) if model in ["cheby1", "ellip"] else 1
            rs = float(self.rs.get()) if model in ["cheby2", "ellip"] else 40
            fir_window = self.firwin.get() if family == "FIR" else None
            iir_struct = self.iir_struct.get() if family == "IIR" else None
            # Validate
            if not fname or not fname.isidentifier():
                raise ValueError("Filter name must be a valid C identifier!")
            if order < 1 or order > 100:
                raise ValueError("Order must be in 1...100!")
            if type(cutoff) is list and (min(cutoff) <= 0 or max(cutoff) >= fs/2):
                raise ValueError("Cutoff must be >0 and < Nyquist.")
            if type(cutoff) is float and (cutoff <= 0 or cutoff >= fs/2):
                raise ValueError("Cutoff must be >0 and < Nyquist.")
            b, a = self._design_filter(ftype, model, order, fs, cutoff, quant, rp, rs, fir_window)
            c_file = self.generate_c_source(fname, b, a, quant, model, fs, cutoff_str, order, ftype, user_header, fir_window, iir_struct)
            self.result.delete(1.0, tk.END)
            self.result.insert(tk.END, c_file)
            self.plot_response(b, a, fs, fname)
            self.plot_impulse_response(b, a, fname)
        except Exception as e:
            messagebox.showerror("Error", f"Error generating filter C code:\n{e}")

    def _design_filter(self, filter_type, model, order, fs, cutoff, quant_type='float', rp=1, rs=40, fir_window=None):
        nyquist = fs / 2.0
        if filter_type in ["lowpass", "highpass"]:
            normalized_cutoff = float(cutoff) / nyquist
        elif filter_type in ["bandpass", "bandstop"]:
            normalized_cutoff = [float(c) / nyquist for c in cutoff]
        else:
            raise ValueError("Unsupported filter type!")
        if model.lower() == "firwin":
            taps = order + 1
            b = signal.firwin(
                taps,
                normalized_cutoff,
                window=fir_window if fir_window else 'hamming',
                pass_zero=(filter_type in ["lowpass", "bandstop"])
            )
            a = np.array([1.0])
        else:
            if model.lower() == "bessel":
                b, a = signal.bessel(order, normalized_cutoff, btype=filter_type, analog=False, norm='phase', output='ba')
            elif model.lower() == "butter":
                b, a = signal.butter(order, normalized_cutoff, btype=filter_type, analog=False, output='ba')
            elif model.lower() == "cheby1":
                b, a = signal.cheby1(order, rp, normalized_cutoff, btype=filter_type, analog=False, output='ba')
            elif model.lower() == "cheby2":
                b, a = signal.cheby2(order, rs, normalized_cutoff, btype=filter_type, analog=False, output='ba')
            elif model.lower() == "ellip":
                b, a = signal.ellip(order, rp, rs, normalized_cutoff, btype=filter_type, analog=False, output='ba')
            else:
                raise ValueError("Unknown filter model!")
        b_quant = quantize_coefficients(b, quant_type)
        a_quant = quantize_coefficients(a, quant_type)
        return b_quant, a_quant

    def generate_c_source(self, filter_name, b, a, quant_type="float", model="fir", fs=1000, cutoff="50", order=8, filter_type="lowpass", user_header="", fir_window=None, iir_struct=None):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        n_b = len(b)
        n_a = len(a)
        window_str = f"\n* Window     : {fir_window}" if fir_window else ""
        struct_str = f"\n* IIR Struct : {iir_struct}" if iir_struct else ""
        header = f"""\
/******************************************************************************
*  Auto-generated Digital Filter C Source
*  {user_header}
*  Generated: {now}
*  Filter Type : {filter_type.upper()}
*  Filter Model: {model.upper()}
*  Order       : {order}
*  Fs (Hz)     : {fs}
*  Cutoff (Hz) : {cutoff}{window_str}{struct_str}
*******************************************************************************/

#include <stddef.h>

#define {filter_name.upper()}_NTAP   ({n_b})
#define {filter_name.upper()}_NA     ({n_a})

"""
        c_b = coefficients_to_c_array(b, f"{filter_name}_b", quant_type)
        if n_a > 1:
            c_a = coefficients_to_c_array(a, f"{filter_name}_a", quant_type)
        else:
            c_a = ""
        fir_func = f"""
/**
 * FIR filter function (Direct Form)
 * y[n] = sum_{filter_name}[i] * x[n - i]
 */
{quant_type} {filter_name}_fir(const {quant_type} *x)
{{
    {quant_type} y = 0;
    for (size_t i = 0; i < {n_b}; ++i)
        y += {filter_name}_b[i] * x[i];
    return y;
}}
""" if n_a == 1 else ""
        iir_func = f"""
/**
 * IIR filter function (Direct Form I, for b and a arrays)
 * y[n] = sum(b[i] * x[n-i]) - sum(a[i] * y[n-i])
 * Assumes a[0] == 1
 */
{quant_type} {filter_name}_iir(const {quant_type} *x, {quant_type} *y_hist)
{{
    {quant_type} y = 0;
    for (size_t i = 0; i < {n_b}; ++i)
        y += {filter_name}_b[i] * x[i];
    for (size_t i = 1; i < {n_a}; ++i)
        y -= {filter_name}_a[i] * y_hist[i - 1];
    return y;
}}
""" if n_a > 1 else ""
        c_file = header + c_b + "\n" + (c_a if c_a else "") + fir_func + iir_func
        return c_file

    def plot_response(self, b, a, fs, fname):
        w, h = signal.freqz(b, a, worN=1024, fs=fs)
        self.ax_mag.clear()
        self.ax_phase.clear()
        self.ax_mag.plot(w, 20 * np.log10(abs(h) + 1e-12), label="Magnitude", color="#0984e3")
        self.ax_mag.set_title("Frequency Response")
        self.ax_mag.set_xlabel("Frequency (Hz)")
        self.ax_mag.set_ylabel("Magnitude (dB)")
        self.ax_mag.grid(True)
        self.ax_phase.plot(w, np.unwrap(np.angle(h)), label="Phase", color="#00b894")
        self.ax_phase.set_ylabel("Phase (radians)")
        self.ax_phase.set_xlabel("Frequency (Hz)")
        self.ax_phase.grid(True)
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_impulse_response(self, b, a, fname):
        n = 50
        impulse = np.zeros(n)
        impulse[0] = 1.0
        y = signal.lfilter(b, a, impulse)
        self.ax_imp.clear()
        self.ax_imp.stem(np.arange(n), y, basefmt=" ", linefmt="#e17055", markerfmt="o")
        self.ax_imp.set_title("Impulse Response")
        self.ax_imp.set_xlabel("Sample")
        self.ax_imp.set_ylabel("Amplitude")
        self.ax_imp.grid(True)
        self.fig_imp.tight_layout()
        self.canvas_imp.draw()

    def export_c_file(self):
        code = self.result.get(1.0, tk.END)
        if not code.strip():
            messagebox.showinfo("Info", "Please generate code first!")
            return
        f = filedialog.asksaveasfile(defaultextension=".c", filetypes=[("C Source", "*.c"), ("All files", "*.*")])
        if f:
            f.write(code)
            f.close()
            messagebox.showinfo("Exported", "C file exported successfully!")

    def copy_c_code(self):
        code = self.result.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(code)
        self.root.update()
        messagebox.showinfo("Copied", "C code copied to clipboard.")

    def show_example_signal(self):
        try:
            ftype = self.filter_type.get()
            model = self.filter_model.get()
            family = self.filter_family.get()
            order = int(self.order.get())
            fs = float(self.fs.get())
            cutoff_raw = self.cutoff.get()
            if ftype in ["bandpass", "bandstop"]:
                cutoff = [float(x.strip()) for x in cutoff_raw.split(",")]
            else:
                cutoff = float(cutoff_raw)
            quant = self.quant.get()
            rp = float(self.rp.get()) if model in ["cheby1", "ellip"] else 1
            rs = float(self.rs.get()) if model in ["cheby2", "ellip"] else 40
            fir_window = self.firwin.get() if family == "FIR" else None
            b, a = self._design_filter(ftype, model, order, fs, cutoff, quant, rp, rs, fir_window)
            t = np.arange(0, 1, 1/fs)
            np.random.seed(0)
            freq = cutoff[0] if isinstance(cutoff, list) else cutoff
            x = 0.6*np.sin(2*np.pi*freq * t) + 0.3*np.random.randn(len(t))
            x[100:200] += 1
            y = signal.lfilter(b, a, x)

            # --- Extra filters (moving average / median) ---
            lines = []
            legends = []

            self.example_fig = Figure(figsize=(8.5,2.4), dpi=98)
            ax = self.example_fig.add_subplot(111)
            ax.plot(t, x, label="Input signal", alpha=0.42, color="#636e72")
            ax.plot(t, y, label="Designed filter", linewidth=2, color="#00b894")
            # Moving average
            if self.show_ma.get():
                try:
                    ma_w = int(self.ma_size.get())
                except Exception:
                    ma_w = 9
                y_ma = moving_average(y, ma_w)
                ax.plot(t, y_ma, label=f"Moving Average (n={ma_w})", linestyle="--", color="#2980b9")
            # Median
            if self.show_med.get():
                try:
                    med_w = int(self.med_size.get())
                except Exception:
                    med_w = 9
                y_med = median_filter(y, med_w)
                ax.plot(t, y_med, label=f"Median Filter (n={med_w})", linestyle="-.", color="#e17055")
            ax.set_xlim([0, 1])
            ax.set_title("Example Signal Filtering (with extra filters)")
            ax.legend()
            ax.grid(True)

            if self.example_canvas is not None:
                self.example_canvas.get_tk_widget().destroy()
            self.example_canvas = FigureCanvasTkAgg(self.example_fig, master=self.right)
            self.example_canvas.get_tk_widget().pack(fill=BOTH, expand=NO, padx=8, pady=5)
            self.example_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Error plotting example signal:\n{e}")

def main():
    app = ttkb.Window(themename="flatly")
    FilterDesignerGUI(app)
    app.mainloop()

if __name__ == "__main__":
    main()
