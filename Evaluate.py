import math
import time

import tkinter as tk
import torch
import numpy as np

from Train import Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

MODEL = '30_0.002327_[450]*8_model.pth'
STATS = '150m stats.npz'

FPS = 120
MSPF = 1000 // FPS

STEPS = 300
DT = 1.0 / STEPS
HALFDT = DT / 2.0
SIXTHDT = DT / 6.0

G = 9.8
M1 = 1.0
M2 = 1.0
M1_M2 = M1 + M2
M1_M2_G = M1_M2 * G

class Visualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Double Pendulum Model Visualizer")
        self.root.geometry("1000x600")

        self.canvas = tk.Canvas(self.root, width=800, height=600, bg='black')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.panel = tk.Frame(self.root, bg="gray12")
        self.panel.pack(side=tk.RIGHT, fill=tk.Y, padx=1, pady=1)

        self.play_btn = tk.Button(self.panel, text="Play", command=self.togglePlay, bg="SeaGreen3")
        self.play_btn.pack(padx=10, pady=5)

        self.loop_btn = tk.Button(self.panel, text="Loop: Off", command=self.toggleLoop, bg="gray50")
        self.loop_btn.pack(padx=10, pady=5)

        self.l1_var = tk.DoubleVar(value=0.75)
        self.l1_var.trace("w", self.resetVars)
        tk.Label(self.panel, text="l1", bg='gray12', fg='white').pack()
        self.l1_entry = tk.Entry(self.panel, textvariable=self.l1_var, bg='gray12', fg='white')
        self.l1_entry.pack(padx=10, pady=5)

        self.l2_var = tk.DoubleVar(value=0.5)
        self.l2_var.trace("w", self.resetVars)
        tk.Label(self.panel, text="l2", bg='gray12', fg='white').pack()
        self.l2_entry = tk.Entry(self.panel, textvariable=self.l2_var, bg='gray12', fg='white')
        self.l2_entry.pack(padx=10, pady=5)

        self.theta1_var = tk.DoubleVar(value=0.0)
        self.theta1_var.trace("w", self.resetVars)
        tk.Label(self.panel, text="theta1 (radians)", bg='gray12', fg='white').pack()
        self.theta1_entry = tk.Entry(self.panel, textvariable=self.theta1_var, bg='gray12', fg='white')
        self.theta1_entry.pack(padx=10, pady=5)

        self.theta2_var = tk.DoubleVar(value=0.0)
        self.theta2_var.trace("w", self.resetVars)
        tk.Label(self.panel, text="theta2 (radians)", bg='gray12', fg='white').pack()
        self.theta2_entry = tk.Entry(self.panel, textvariable=self.theta2_var, bg='gray12', fg='white')
        self.theta2_entry.pack(padx=10, pady=5)

        self.nsteps_var = tk.IntVar(value=300)
        self.nsteps_var.trace("w", self.resetVars)
        tk.Label(self.panel, text="nsteps", bg='gray12', fg='white').pack()
        self.nsteps_entry = tk.Entry(self.panel, textvariable=self.nsteps_var, bg='gray12', fg='white')
        self.nsteps_entry.pack(padx=10, pady=5)

        self.fps_label = tk.Label(self.canvas, text="FPS: 0.0", font=("Arial", 12), anchor="nw", bg="black", fg="white")
        self.fps_label.pack(padx=10, pady=5)

        self.is_playing = False
        self.reset = False
        self.is_loop = False
        self.current_step = 0
        self.last_time = None
        self.current_fps = 0.0
        self.scale = 200
        self.center_x = 400
        self.center_y = 300
        self.s = [0.0, 0.0, 0.0, 0.0]
        self.consts = None
        self.l1 = 0.75
        self.l2 = 0.5
        self.nsteps = 300
        self.mse = 0.0
        self.x_pred = 0.0
        self.y_pred = 0.0
        self.x_true = 0.0
        self.y_true = 0.0
        self.x_final = 0.0
        self.y_final = 0.0
        self.resetVars()

        stats = np.load(STATS)
        self.input_mean, self.input_std = stats['input_mean'], stats['input_std']
        self.target_mean, self.target_std = stats['target_mean'], stats['target_std']
        del stats


        self.model = Predictor()
        self.model.load_state_dict(torch.load(MODEL, map_location=device))
        self.model.eval()

        self.updatePred()

        self.canvas.bind("<Configure>", self.resize)
        self.animate()

        self.root.mainloop()

    def resize(self, event=None):
        width = event.width if event else self.canvas.winfo_width()
        height = event.height if event else self.canvas.winfo_height()

        self.center_x = width / 2
        self.center_y = height / 2
        self.canvas.delete('del2')
        self.canvas.create_oval(self.center_x - 5,  self.center_y - 5, self.center_x + 5,  self.center_y + 5, fill='white', tags='del2')

        margin = 20
        max_length = self.l1 + self.l2
        self.scale = (min(width, height) / 2 - margin) / (max_length)

    def togglePlay(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.config(text="Pause", bg="dark orange")
            self.disableEdit()
            if self.current_step == 0:  # Only reset consts if starting fresh
                self.l1 = self.l1_var.get()
                self.l2 = self.l2_var.get()
                self.nsteps = self.nsteps_var.get()
        else:
            self.play_btn.config(text="Play", bg="SeaGreen3")
            self.enableEdit()

    def toggleLoop(self):
        self.is_loop = not self.is_loop
        self.loop_btn.config(text=f"Loop: {'On' if self.is_loop else 'Off'}", 
                             bg=f"{'SteelBlue3' if self.is_loop else 'gray50'}")

    def disableEdit(self):
        self.l1_entry.config(state='disabled')
        self.l2_entry.config(state='disabled')
        self.theta1_entry.config(state='disabled')
        self.theta2_entry.config(state='disabled')
        self.nsteps_entry.config(state='disabled')

    def enableEdit(self):
        self.l1_entry.config(state='normal')
        self.l2_entry.config(state='normal')
        self.theta1_entry.config(state='normal')
        self.theta2_entry.config(state='normal')
        self.nsteps_entry.config(state='normal')

    def computeConsts(self):
        l1, l2 = self.l1, self.l2
        return {
            'm_total_g': M1_M2_G,
            'M2_l1': M2 * l1,
            'M2_l2': M2 * l2,
            'm_total_l': M1_M2 * l1,
            'l1': l1,
            'l2': l2,
            'l2_div_l1': l2 / l1
        }

    def resetVars(self, *args):
        if not self.is_playing or self.reset:
            try:
                self.s = [self.theta1_var.get(), 0.0, self.theta2_var.get(), 0.0]
                self.l1 = self.l1_var.get()
                self.l2 = self.l2_var.get()
                self.resize()
                self.nsteps = self.nsteps_var.get()
                self.consts = self.computeConsts()
                self.current_step = 0
                self.updatePred()
            except Exception as e:
                print("Invalid var", e)

    def updatePred(self):
        try:
            theta1 = self.theta1_var.get()
            theta2 = self.theta2_var.get()
            l1 = self.l1_var.get()
            l2 = self.l2_var.get()
            nsteps = self.nsteps_var.get()
            t = nsteps / STEPS
            sin1 = math.sin(theta1)
            cos1 = math.cos(theta1)
            sin2 = math.sin(theta2)
            cos2 = math.cos(theta2)
            inputs = np.array([sin1, cos1, sin2, cos2, l1, l2, t], dtype=np.float32)
            norm_inputs = (inputs - self.input_mean) / self.input_std
            tensor = torch.from_numpy(norm_inputs).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_norm = self.model(tensor)
            pred = pred_norm.squeeze(0).numpy() * self.target_std + self.target_mean
            self.x_pred, self.y_pred = pred
        except Exception as e:
            print("Skipped prediction", e)
            

    def clampAngle(self, theta):
        return theta - 2.0 * math.pi * round(theta / (2.0 * math.pi))

    def derivs(self, s):
        si = math.sin(s[2] - s[0])
        c = math.cos(s[2] - s[0])
        sin_s2 = math.sin(s[2])
        m_total_g_sin_s0 = self.consts['m_total_g'] * math.sin(s[0])
        M2_l2_s3_squared_si = self.consts['M2_l2'] * s[3] * s[3] * si
        s1_squared_si = s[1] * s[1] * si
        den1 = self.consts['m_total_l'] - self.consts['M2_l1'] * c * c

        dsdt0 = s[1]
        dsdt1 = (
            self.consts['M2_l1'] * s1_squared_si * c +
            M2 * G * sin_s2 * c +
            M2_l2_s3_squared_si -
            m_total_g_sin_s0
        ) / den1

        dsdt2 = s[3]
        dsdt3 = (
            - M2_l2_s3_squared_si * c +
            m_total_g_sin_s0 * c -
            self.consts['m_total_l'] * s1_squared_si -
            self.consts['m_total_g'] * sin_s2
        ) / (self.consts['l2_div_l1'] * den1)

        return [dsdt0, dsdt1, dsdt2, dsdt3]

    def RK4Step(self):
        k1 = self.derivs(self.s)
        tmp = [self.s[i] + HALFDT * k1[i] for i in range(4)]
        k2 = self.derivs(tmp)
        tmp = [self.s[i] + HALFDT * k2[i] for i in range(4)]
        k3 = self.derivs(tmp)
        tmp = [self.s[i] + DT * k3[i] for i in range(4)]
        k4 = self.derivs(tmp)
        for i in range(4):
            self.s[i] += SIXTHDT * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

    def animate(self):
        now = time.time()
        if self.last_time is not None:
            dt = now - self.last_time
            if dt > 0:
                self.current_fps = 1.0 / dt
                self.fps_label.config(text=f"FPS: {self.current_fps:.1f}")
        self.last_time = now

        self.draw()
        self.root.after(MSPF, self.animate)


    def draw(self):
        self.canvas.delete('del')

        if not self.is_playing:
            try:
                theta1 = self.s[0]
                theta2 = self.s[2]
                l1 = self.l1
                l2 = self.l2
            except:
                return
            x1 = l1 * math.sin(theta1)
            y1 = -l1 * math.cos(theta1)
            x2 = x1 + l2 * math.sin(theta2)
            y2 = y1 - l2 * math.cos(theta2)
            self.x_true = x2
            self.y_true = y2
        else:

            if self.reset:
                self.resetVars()
                self.reset = False

            steps_per_frame = max(1, STEPS // FPS)  # 5
            advanced = 0
            for _ in range(steps_per_frame):
                if self.current_step >= self.nsteps:
                    break
                self.RK4Step()
                self.current_step += 1
                advanced += 1
                if self.current_step % STEPS == 0:
                    self.s[0] = self.clampAngle(self.s[0])
                    self.s[2] = self.clampAngle(self.s[2])

            theta1 = self.s[0]
            theta2 = self.s[2]
            x1 = self.l1 * math.sin(theta1)
            y1 = -self.l1 * math.cos(theta1)
            x2 = x1 + self.l2 * math.sin(theta2)
            y2 = y1 - self.l2 * math.cos(theta2)
            self.x_true = x2
            self.y_true = y2

            if advanced < steps_per_frame and self.current_step >= self.nsteps:
                self.s[0] = self.clampAngle(self.s[0])
                self.s[2] = self.clampAngle(self.s[2])
                print(x2)
                self.x_final = x2
                self.y_final = y2
                if self.is_loop:
                    self.current_step = 0
                    self.s = [self.theta1_var.get(), 0.0, self.theta2_var.get(), 0.0]
                    self.l1 = self.l1_var.get()
                    self.l2 = self.l2_var.get()
                    self.nsteps = self.nsteps_var.get()
                else:
                    self.togglePlay()
                    self.reset = True
            else:
                self.x_final = self.x_true
                self.y_final = self.y_true

        self.mse = ((self.x_pred - self.x_true) ** 2 + (self.y_pred - self.y_true) ** 2) / 2

        ox, oy = self.center_x, self.center_y
        x1_s = ox + x1 * self.scale
        y1_s = oy - y1 * self.scale
        x2_s = ox + x2 * self.scale
        y2_s = oy - y2 * self.scale

        self.canvas.create_line(ox, oy, x1_s, y1_s, width=3, fill='blue', tags='del')

        self.canvas.create_oval(x1_s - 10, y1_s - 10, x1_s + 10, y1_s + 10, fill='blue', tags='del')

        self.canvas.create_line(x1_s, y1_s, x2_s, y2_s, width=3, fill='green', tags='del')

        self.canvas.create_oval(x2_s - 10, y2_s - 10, x2_s + 10, y2_s + 10, fill='green', tags='del')

        x_final = ox + self.x_final * self.scale
        y_final = oy - self.y_final * self.scale
        self.canvas.create_oval(x_final - 10, y_final - 10, x_final + 10, y_final + 10, outline='green', width=4, tags='del')

        x_pred_s = ox + self.x_pred * self.scale
        y_pred_s = oy - self.y_pred * self.scale
        self.canvas.create_oval(x_pred_s - 10, y_pred_s - 10, x_pred_s + 10, y_pred_s + 10, outline='red', width=2, tags='del')

        self.canvas.create_text(10, self.canvas.winfo_height() - 30, anchor='sw', text=f"MSE: {self.mse:.6f}", font=("Arial", 12), fill='white', tags='del')

if __name__ == "__main__":
    Visualizer()
