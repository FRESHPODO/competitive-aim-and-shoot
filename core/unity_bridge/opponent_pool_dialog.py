"""Editable ordered opponent list for the training UI."""
from copy import deepcopy
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from unity_bridge.opponent_pool import normalize
from unity_bridge.player_presets import PRESETS, MODES, LABEL_TO_ID

KEEP = '화면 설정 유지'


def edit(parent, config, on_save):
    entries = deepcopy(config.get('opponents', []))
    window = tk.Toplevel(parent)
    window.title('B 고정 상대 목록 · A만 학습')
    window.geometry('1050x540')
    window.transient(parent)
    ttk.Label(window, text='위에서 아래 순서로 경기마다 교대합니다. 마지막 다음에는 첫 상대부터 반복합니다. B는 학습하지 않습니다.',
              padding=10).pack(anchor='w')
    table = ttk.Treeview(window, columns=('name', 'preset', 'weapon', 'model'), show='headings', height=9)
    for key, title, width in [('name', '상대 이름', 140), ('preset', '플레이어', 160),
                              ('weapon', '무기', 110), ('model', 'PPO 모델', 550)]:
        table.heading(key, text=title)
        table.column(key, width=width)
    table.pack(fill='both', expand=True, padx=10)
    name = tk.StringVar()
    model = tk.StringVar()
    preset = tk.StringVar(value=KEEP)
    weapon = tk.StringVar(value=KEEP)
    form = ttk.Frame(window, padding=10)
    form.pack(fill='x')
    for column, (title, var, values) in enumerate([
        ('이름', name, None), ('플레이어', preset, [KEEP] + list(MODES[1:])),
        ('무기', weapon, [KEEP, '기본형', 'AR형', '저격형'])]):
        ttk.Label(form, text=title).grid(row=0, column=column*2, padx=4)
        if values:
            ttk.Combobox(form, textvariable=var, values=values, state='readonly', width=22).grid(row=0, column=column*2+1)
        else:
            ttk.Entry(form, textvariable=var, width=20).grid(row=0, column=column*2+1)
    ttk.Label(form, text='PPO 모델').grid(row=1, column=0, pady=8)
    ttk.Entry(form, textvariable=model, width=100).grid(row=1, column=1, columnspan=4, sticky='ew')

    def choose():
        path = filedialog.askopenfilename(parent=window, title='고정 상대 PPO 모델', filetypes=[('PPO checkpoint', '*.zip')])
        if path:
            model.set(path)
            if not name.get():
                name.set(Path(path).parent.name)
    ttk.Button(form, text='모델 선택', command=choose).grid(row=1, column=5, padx=4)

    def refresh(selected=None):
        table.delete(*table.get_children())
        for i, entry in enumerate(entries):
            table.insert('', 'end', iid=str(i), values=(entry['name'],
                PRESETS.get(entry.get('presetId'), {}).get('label', KEEP), entry.get('weaponPreset') or KEEP, entry['model']))
        if selected is not None and 0 <= selected < len(entries):
            table.selection_set(str(selected))
            table.see(str(selected))

    def selection():
        selected = table.selection()
        return int(selected[0]) if selected else None

    def load(_=None):
        index = selection()
        if index is not None:
            entry = entries[index]
            name.set(entry['name']); model.set(entry['model'])
            preset.set(PRESETS.get(entry.get('presetId'), {}).get('label', KEEP))
            weapon.set(entry.get('weaponPreset') or KEEP)
    table.bind('<<TreeviewSelect>>', load)

    def store(replace=False):
        entry = dict(name=name.get().strip(), model=model.get().strip(),
                     presetId=LABEL_TO_ID.get(preset.get()), weaponPreset=None if weapon.get()==KEEP else weapon.get())
        try:
            entry = normalize(dict(enabled=True, opponents=[entry]))['opponents'][0]
        except Exception as error:
            messagebox.showerror('상대 설정 오류', str(error), parent=window)
            return
        index = selection() if replace else None
        if replace and index is None:
            return
        if index is None:
            entries.append(entry); index=len(entries)-1
        else:
            entries[index]=entry
        refresh(index)

    def remove():
        index = selection()
        if index is not None:
            entries.pop(index); refresh(min(index,len(entries)-1))

    def move(delta):
        index = selection()
        if index is not None and 0 <= index+delta < len(entries):
            entries[index],entries[index+delta]=entries[index+delta],entries[index]
            refresh(index+delta)

    def save():
        # An empty list can be saved while disabled; enabling requires a member.
        on_save(dict(config, learnerSide='A', opponents=deepcopy(entries)))
        window.destroy()

    buttons = ttk.Frame(window, padding=10)
    buttons.pack(fill='x')
    for title, callback in [('추가',lambda:store()), ('선택 수정',lambda:store(True)),
                             ('선택 삭제',remove), ('위로',lambda:move(-1)), ('아래로',lambda:move(1))]:
        ttk.Button(buttons,text=title,command=callback).pack(side='left',padx=3)
    ttk.Button(buttons,text='목록 저장',command=save).pack(side='right')
    ttk.Label(window,text='플레이어 프리셋은 해당 상대의 8개 파라미터를 적용합니다. 지정하지 않은 조건과 B 보상은 본 화면 설정을 유지합니다.',
              padding=10).pack(anchor='w')
    refresh()
    return window
