"""Shared alpha editor with live, signed terminal-reward calculations."""
import tkinter as tk
from tkinter import ttk
from unity_bridge.competitive_rewards import preview_text


def build_reward_editor(parent,fields,*,factor_label=None):
    for column,title in enumerate(('방식','고정값','최소','최대','단위')):
        ttk.Label(parent,text=title).grid(row=0,column=column,padx=1)
        parent.columnconfigure(column,weight=1)
    ttk.Label(parent,text=factor_label or '보상 비율 α (0~1)').grid(row=1,column=0,columnspan=5,sticky='w',pady=(8,2))
    ttk.Combobox(parent,textvariable=fields['mode'],values=['고정','범위'],width=4,state='readonly').grid(row=2,column=0,sticky='ew')
    entries=[]
    for column,key in enumerate(('value','min','max','step'),1):
        entry=ttk.Entry(parent,textvariable=fields[key],width=5)
        entry.grid(row=2,column=column,sticky='ew',padx=1);entries.append(entry)
    preview=None
    if factor_label is None:
        preview=tk.StringVar(master=parent)
        ttk.Label(parent,textvariable=preview,font=('Malgun Gothic',11,'bold'),wraplength=240,justify='left').grid(row=3,column=0,columnspan=5,sticky='w',pady=(10,4))
        ttk.Label(parent,text='성공 +2α · 실패 −2(1−α)\n무승부 0 · 확률이 아닌 보상값',wraplength=240).grid(row=4,column=0,columnspan=5,sticky='w',pady=(2,8))
    def refresh(*_):
        fixed=fields['mode'].get()=='고정'
        entries[0].configure(state='normal' if fixed else 'disabled')
        for entry in entries[1:]:entry.configure(state='disabled' if fixed else 'normal')
        if preview is None:return
        try:
            spec=dict(mode='fixed' if fixed else 'uniform')
            for key in (('value',) if fixed else ('min','max','step')):spec[key]=float(fields[key].get())
            preview.set(preview_text(spec))
        except (ValueError,TypeError,KeyError):preview.set('α는 0~1, 추출 단위는 양수로 입력하세요.')
    for variable in fields.values():variable.trace_add('write',refresh)
    refresh()
    return preview
