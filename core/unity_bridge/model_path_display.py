"""Copyable paths for the models produced by the selected training run."""
import tkinter as tk
from tkinter import ttk

from unity_bridge.storage import model_directory


def build_model_paths(parent, model_root, files):
    panel = ttk.Frame(parent)
    panel.pack(fill='x', pady=3)
    panel.columnconfigure(1, weight=1)
    variables = []
    for row, (label, _) in enumerate(files):
        var = tk.StringVar(master=panel)
        variables.append(var)
        ttk.Label(panel, text=label, width=12).grid(row=row, column=0, sticky='w')
        ttk.Entry(panel, textvariable=var, state='readonly').grid(row=row, column=1, sticky='ew')

        def copy_path(value=var):
            panel.clipboard_clear()
            panel.clipboard_append(value.get())

        ttk.Button(panel, text='경로 복사', command=copy_path).grid(row=row, column=2, padx=(4, 0))

    def refresh(run):
        if run and (run/'training.json').is_file():
            folder = model_directory(run)
        else:
            folder = model_root/run.name if run else model_root
        for var, (_, filename) in zip(variables, files):
            value = str(folder/filename if run else folder)
            if var.get() != value:
                var.set(value)

    refresh(None)
    return refresh
