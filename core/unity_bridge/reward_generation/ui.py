"""Separate desktop UI for a single conditional learner and recent generations."""
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from unity_bridge.reward_generation.config import CUSTOM, KIND, defaults, validate
from unity_bridge.player_presets import PRESETS, MODES, LABEL_TO_ID
from unity_bridge.training_settings import AGENT, GAME, WEAPON
from unity_bridge.reward_editor import build_reward_editor
from unity_bridge.model_path_display import build_model_paths
from unity_bridge.trainer_state import load_state, save_state
from unity_bridge.storage import ROOT, model_directory, training_directory

from unity_bridge.reward_generation.paths import RESULTS, PREFERENCES, MODELS as MODEL_ROOT, LOGS, REQUESTS


class GenerationTrainer:
    def __init__(self, root, *, state_path=PREFERENCES):
        self.root = root
        self.state_path = Path(state_path)
        self.process = None
        self.processes = {}
        self.poll_job = None
        self.run = None
        self.save_job = None
        self.restoring = True
        self.inputs = {}
        self.controls = {}
        self.preset_lists = {}
        self.preset_counts = {}
        self.custom = {side: tk.BooleanVar(root, value=False) for side in 'AB'}
        self.reward_vars = {}
        self.reward_previews = {}
        self.resume = tk.StringVar(root)
        self.status = tk.StringVar(root, value='학습자 PPO 하나만 갱신 · 상대는 최근 세대에서 무작위 선택')
        self.generation_status = tk.StringVar(root, value='최초 상대: 무보류 · 첫 세대 등록 후 자동 제거')
        self.job = tk.StringVar(root)
        self.preset_ids = [LABEL_TO_ID[label] for label in MODES if label in LABEL_TO_ID]
        root.title('단일 PPO · 피해·Critical 세대 학습기')
        root.geometry(f'{min(1860, root.winfo_screenwidth()-40)}x{min(1030, root.winfo_screenheight()-70)}')
        root.minsize(1380, 800)
        shell = ttk.Frame(root, padding=12)
        shell.pack(fill='both', expand=True)
        ttk.Label(shell, text='단일 PPO · 피해·Critical 세대 학습', font=('Malgun Gothic', 17, 'bold')).pack(anchor='w')
        ttk.Label(shell, text='내 조준 프리셋 · 내 α/피해/Critical 계수 · 공통 게임 조건을 바꿔 하나의 모델을 학습 / B는 학습하지 않음').pack(anchor='w')
        c = defaults()
        rows = [
            [('episodes', '총 학습 경기 수'), ('timeout', '경기 제한 (초)'), ('workers', '병렬 경기 수 (0: 자동)'), ('batchEpisodes', '갱신당 경기 수')],
            [('generationGames', '세대 교체 주기 (경기)'), ('retainedGenerations', '최근 상대 세대 수'), ('checkpointSeconds', '복구 저장 간격 (초)'), ('seed', 'Seed (빈칸: 랜덤)')],
        ]
        for specs in rows:
            row = ttk.Frame(shell)
            row.pack(fill='x', pady=4)
            for key, label in specs:
                ttk.Label(row, text=label).pack(side='left', padx=(0, 5))
                variable = tk.StringVar(root, value='' if c[key] is None else str(c[key]))
                self.controls[key] = variable
                if key == 'retainedGenerations':
                    ttk.Combobox(row, textvariable=variable, values=['2', '3'], state='readonly', width=4).pack(side='left', padx=(0, 18))
                else:
                    ttk.Entry(row, textvariable=variable, width=11).pack(side='left', padx=(0, 18))
        row = ttk.Frame(shell)
        row.pack(fill='x', pady=4)
        ttk.Label(row, text='이어 학습 폴더').pack(side='left')
        ttk.Entry(row, textvariable=self.resume, width=72, state='readonly').pack(side='left', padx=6)
        ttk.Button(row, text='선택', command=self.select_resume).pack(side='left')
        ttk.Button(row, text='새 학습으로 전환', command=self.clear_resume).pack(side='left', padx=6)
        ttk.Label(row, text='CPU 병렬 · 리플레이 자동 기록 안 함').pack(side='left', padx=12)
        area = ttk.Frame(shell)
        area.pack(fill='both', expand=True, pady=6)
        canvas = tk.Canvas(area, highlightthickness=0)
        scroll = ttk.Scrollbar(area, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        panels = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=panels, anchor='nw')
        panels.bind('<Configure>', lambda _: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(window, width=e.width))
        titles = ['학습자 A · 조준 조건', '학습자 A · PPO 보상', '공통 게임 조건', '상대 B · PPO 보상', '상대 B · 조준 조건']
        self.panels = []
        for column, title in enumerate(titles):
            panels.columnconfigure(column, weight=1, uniform='conditions')
            frame = ttk.LabelFrame(panels, text=title, padding=8)
            frame.grid(row=0, column=column, sticky='nsew', padx=3)
            self.panels.append(frame)
        for side, column in [('A', 0), ('B', 4)]:
            frame = self.panels[column]
            ttk.Label(frame, text='선택한 후보 중 매 경기 하나를 균등 추출', wraplength=260).pack(anchor='w')
            tabs = ttk.Notebook(frame)
            tabs.pack(fill='both', expand=True, pady=5)
            presets = ttk.Frame(tabs)
            custom = ttk.Frame(tabs)
            tabs.add(presets, text='프리셋 선택')
            tabs.add(custom, text='커스텀 범위')
            box = tk.Listbox(presets, selectmode=tk.MULTIPLE, exportselection=False, height=18)
            for pid in self.preset_ids:
                box.insert('end', PRESETS[pid]['label'])
            box.pack(fill='both', expand=True)
            self.preset_lists[side] = box
            box.bind('<<ListboxSelect>>', lambda _, s=side: self.presets_changed(s))
            for label, mode in [('프로 원값 전체', 'professional'), ('아마추어 원값 전체', 'amateur'), ('평균 2개', 'means'), ('선택 해제', 'none')]:
                ttk.Button(presets, text=label, command=lambda s=side, m=mode: self.select_group(s, m)).pack(fill='x', pady=1)
            self.preset_counts[side] = tk.StringVar(root)
            ttk.Label(presets, textvariable=self.preset_counts[side]).pack(anchor='w', pady=3)
            ttk.Checkbutton(custom, text='커스텀 범위도 추출 후보에 포함', variable=self.custom[side], command=lambda s=side: self.presets_changed(s)).pack(anchor='w')
            self.parameter_table(custom, AGENT, side + '.')
            ttk.Label(frame, text='프리셋 선택 시 원본 조준 모델의 8개 값 전체 적용. 추가 PPO 보상과 공통 조건은 별도.', wraplength=260).pack(anchor='w', pady=5)
        for side, column in [('A', 1), ('B', 3)]:
            frame = self.panels[column]
            fields = {key: tk.StringVar(root) for key in ('mode', 'value', 'min', 'max', 'step')}
            self.reward_vars[side] = fields
            editor = ttk.Frame(frame)
            editor.pack(fill='x')
            self.reward_previews[side] = build_reward_editor(editor, fields)
            self.parameter_table(frame, [('damageFactor', '누적 피해 보상 계수 (0~1)'), ('criticalFactor', 'Critical 처치 보상 (0~1)')], side + '.')
            note = ('이 모델 하나만 추가 학습합니다.' if side == 'A' else
                    'B는 경기마다 최근 세대의 저장 모델을 사용합니다. 학습으로 갱신하는 모델은 A뿐입니다.\n상대 α·피해·Critical 계수는 매 경기 추출하지만 A에 공개하지 않습니다.\n무보류 상대 기간에는 α와 무관하게 요청을 승인합니다.')
            ttk.Label(frame, text=note, wraplength=260).pack(anchor='w', pady=12)
        self.parameter_table(self.panels[2], GAME + [x for x in WEAPON if x[0] != 'spread'])
        self.instant = tk.BooleanVar(root, value=True)
        ttk.Checkbutton(self.panels[2], text='중앙 치명타 즉사', variable=self.instant).pack(anchor='w', pady=6)
        ttk.Label(self.panels[2], text='양쪽에 같은 공격 조건 적용\n탄 퍼짐 없음\n범위: 최소 + k × 추출 단위 (최대 이하)\n각 경기 동안 조건 고정', wraplength=260).pack(anchor='w', pady=10)
        ttk.Label(shell, textvariable=self.generation_status, font=('Malgun Gothic', 11, 'bold')).pack(anchor='w', pady=4)
        buttons = ttk.Frame(shell)
        buttons.pack(fill='x', pady=4)
        for label, action in [('설정 저장', self.export_config), ('설정 불러오기', self.import_config),
                              ('새 학습 시작', self.start_new), ('이어서 학습', self.start_resume),
                              ('선택 작업 중단·저장', self.stop), ('모델 폴더', self.open_models)]:
            ttk.Button(buttons, text=label, command=action).pack(side='left', padx=3)
        self.refresh_model_paths = build_model_paths(shell, MODEL_ROOT, [('최신 모델', 'learner.zip')])
        jobs = ttk.Frame(shell)
        jobs.pack(fill='x', pady=4)
        ttk.Label(jobs, text='표시할 학습 작업').pack(side='left')
        self.job_combo = ttk.Combobox(jobs, textvariable=self.job, state='readonly', width=47)
        self.job_combo.pack(side='left', padx=5)
        self.job_combo.bind('<<ComboboxSelected>>', self.select_job)
        ttk.Button(jobs, text='목록 갱신', command=self.refresh_jobs).pack(side='left')
        self.bar = ttk.Progressbar(shell, maximum=100)
        self.bar.pack(fill='x', pady=4)
        ttk.Label(shell, textvariable=self.status, wraplength=1450).pack(anchor='w')
        self.log = tk.Text(shell, height=4, state='disabled', font=('Consolas', 9))
        self.log.pack(fill='x', pady=3)
        self.apply_config(c)
        saved = load_state(self.state_path)
        if saved:
            self.restore_draft(saved)
        self.restoring = False
        for variable in [*self.controls.values(), self.resume, self.instant, *self.custom.values(),
                         *[v for group in self.inputs.values() for v in group.values()],
                         *[v for group in self.reward_vars.values() for v in group.values()]]:
            variable.trace_add('write', self.schedule_save)
        self.refresh_jobs()
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.poll()

    def parameter_table(self, parent, specs, prefix=''):
        table = ttk.Frame(parent)
        table.pack(fill='x')
        for col, label in enumerate(('방식', '고정', '최소', '최대', '단위')):
            ttk.Label(table, text=label).grid(row=0, column=col)
            table.columnconfigure(col, weight=1)
        for index, (key, label, *_limits) in enumerate(specs):
            label = label.replace('원본 조준 성공 조건 (PPO 보상 아님)', '원본 조준 성공 보상').replace('원본 조준 실패 조건 (PPO 비용 아님)', '원본 조준 실패 비용')
            ttk.Label(table, text=label, wraplength=255).grid(row=2*index+1, column=0, columnspan=5, sticky='w', pady=(6, 1))
            fields = {name: tk.StringVar(self.root) for name in ('mode', 'value', 'min', 'max', 'step')}
            self.inputs[prefix+key] = fields
            ttk.Combobox(table, textvariable=fields['mode'], values=['고정', '범위'], width=4, state='readonly').grid(row=2*index+2, column=0, sticky='ew')
            entries = []
            for col, name in enumerate(('value', 'min', 'max', 'step'), 1):
                entry = ttk.Entry(table, textvariable=fields[name], width=5)
                entry.grid(row=2*index+2, column=col, sticky='ew', padx=1)
                entries.append(entry)
            def toggle(*_, f=fields, e=entries):
                fixed = f['mode'].get() == '고정'
                e[0].configure(state='normal' if fixed else 'disabled')
                for entry in e[1:]:
                    entry.configure(state='disabled' if fixed else 'normal')
            fields['mode'].trace_add('write', toggle)
            toggle()

    def presets_changed(self, side):
        count = len(self.preset_lists[side].curselection()) + int(self.custom[side].get())
        self.preset_counts[side].set(f'추출 후보 {count}개 · 각 후보 동일 확률')
        self.schedule_save()

    def select_group(self, side, mode):
        box = self.preset_lists[side]
        box.selection_clear(0, 'end')
        for index, pid in enumerate(self.preset_ids):
            p = PRESETS[pid]
            if (mode == 'means' and p['kind'] == 'group_mean') or (mode in ('professional', 'amateur') and p['group'] == mode and p['kind'] == 'individual_condition'):
                box.selection_set(index)
        self.presets_changed(side)

    @staticmethod
    def spec(fields):
        fixed = fields['mode'].get() == '고정'
        required = ('value',) if fixed else ('min', 'max', 'step')
        result = dict(mode='fixed' if fixed else 'uniform')
        for key in required:
            result[key] = float(fields[key].get())
        for key in ('value', 'min', 'max', 'step'):
            if key not in result:
                try:
                    result[key] = float(fields[key].get())
                except ValueError:
                    result[key] = 1. if key == 'step' else result.get('value', result.get('min', 0.))
        return result

    def configuration(self):
        c = defaults()
        for key, variable in self.controls.items():
            value = variable.get().strip()
            c[key] = (int(value) if value else None) if key == 'seed' else float(value) if key in ('timeout', 'checkpointSeconds') else int(value)
        for key, fields in self.inputs.items():
            if key.split('.')[-1] in ('damageFactor', 'criticalFactor'):
                side, factor = key.split('.')
                c['rewardConfig'][side][factor] = self.spec(fields)
            elif key in ('minDamage', 'maxDamage', 'cooldown'):
                for side in 'AB':
                    c['settings'][side+'.'+key] = self.spec(fields)
            else:
                c['settings'][key] = self.spec(fields)
        for side in 'AB':
            c['settings'][side+'.instantKill'].update(value=int(self.instant.get()), min=int(self.instant.get()), max=int(self.instant.get()))
            c['presetChoices'][side] = [self.preset_ids[i] for i in self.preset_lists[side].curselection()]
            if self.custom[side].get():
                c['presetChoices'][side].append(CUSTOM)
            c['rewardConfig'][side]['alpha'] = self.spec(self.reward_vars[side])
        return validate(c)

    def apply_config(self, config):
        c = validate(config)
        for key, variable in self.controls.items():
            variable.set('' if c[key] is None else str(c[key]))
        for key, fields in self.inputs.items():
            if key.split('.')[-1] in ('damageFactor', 'criticalFactor'):
                side, factor = key.split('.')
                spec = c['rewardConfig'][side][factor]
            else:
                spec = c['settings']['A.'+key if key in ('minDamage', 'maxDamage', 'cooldown') else key]
            self.set_fields(fields, spec)
        self.instant.set(bool(c['settings']['A.instantKill']['value']))
        for side in 'AB':
            self.set_fields(self.reward_vars[side], c['rewardConfig'][side]['alpha'])
            box = self.preset_lists[side]
            box.selection_clear(0, 'end')
            for pid in c['presetChoices'][side]:
                if pid != CUSTOM and pid not in self.preset_ids:
                    self.preset_ids.append(pid)
                    for other in self.preset_lists.values():
                        other.insert('end', PRESETS[pid]['label'])
                if pid != CUSTOM:
                    box.selection_set(self.preset_ids.index(pid))
            self.custom[side].set(CUSTOM in c['presetChoices'][side])
            self.presets_changed(side)

    @staticmethod
    def set_fields(fields, spec):
        for key, variable in fields.items():
            variable.set(('고정' if spec[key] == 'fixed' else '범위') if key == 'mode' else str(spec[key]))

    def draft(self):
        return dict(controls={k: v.get() for k, v in self.controls.items()},
            inputs={k: {n: v.get() for n, v in f.items()} for k, f in self.inputs.items()},
            rewards={s: {n: v.get() for n, v in f.items()} for s, f in self.reward_vars.items()},
            presetChoices={s: [self.preset_ids[i] for i in box.curselection()] for s, box in self.preset_lists.items()},
            custom={s: v.get() for s, v in self.custom.items()}, instant=self.instant.get(),
            resume=self.resume.get(), lastRun=str(self.run) if self.run else None)

    def restore_draft(self, draft):
        for key, text in draft.get('controls', {}).items():
            if key in self.controls:
                self.controls[key].set(text)
        for container, name in ((self.inputs, 'inputs'), (self.reward_vars, 'rewards')):
            for key, fields in draft.get(name, {}).items():
                if key in container:
                    for field, text in fields.items():
                        if field in container[key]:
                            container[key][field].set(text)
        for side in 'AB':
            box = self.preset_lists[side]
            box.selection_clear(0, 'end')
            for pid in draft.get('presetChoices', {}).get(side, []):
                if pid in PRESETS and pid not in self.preset_ids:
                    self.preset_ids.append(pid)
                    for other in self.preset_lists.values():
                        other.insert('end', PRESETS[pid]['label'])
                if pid in self.preset_ids:
                    box.selection_set(self.preset_ids.index(pid))
            self.custom[side].set(draft.get('custom', {}).get(side, False))
            self.presets_changed(side)
        self.instant.set(draft.get('instant', True))
        self.resume.set(draft.get('resume', ''))
        if draft.get('lastRun'):
            self.run = Path(draft['lastRun'])

    def schedule_save(self, *_):
        if self.restoring:
            return
        if self.save_job is not None:
            self.root.after_cancel(self.save_job)
        self.save_job = self.root.after(400, self.persist)

    def persist(self):
        self.save_job = None
        save_state(self.state_path, self.draft())

    def export_config(self):
        try:
            config = self.configuration()
            path = filedialog.asksaveasfilename(parent=self.root, defaultextension='.json', filetypes=[('JSON', '*.json')])
            if path:
                save_state(path, config)
        except (ValueError, OSError) as error:
            messagebox.showerror('설정 확인', str(error), parent=self.root)

    def import_config(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[('JSON', '*.json')])
        if path:
            try:
                self.apply_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))
                self.clear_resume()
                self.persist()
            except (ValueError, OSError) as error:
                messagebox.showerror('설정 확인', str(error), parent=self.root)

    def select_resume(self):
        path = filedialog.askdirectory(parent=self.root, initialdir=RESULTS)
        if path:
            try:
                records = training_directory(path)
                meta = load_state(records / 'training.json') or {}
                if meta.get('trainerKind') != KIND:
                    raise ValueError('이 학습기의 작업 기록 또는 모델 폴더를 선택하세요.')
                self.apply_config(meta['config'])
                self.resume.set(str(records))
                self.run = records
                self.refresh_model_paths(self.run)
                self.status.set('기존 목표·설정·상대 세대 목록을 유지해 이어 학습합니다. 목표 경기 수만 늘릴 수 있습니다.')
                self.persist()
            except (ValueError, OSError) as error:
                messagebox.showerror('이어 학습 확인', str(error), parent=self.root)

    def clear_resume(self):
        self.resume.set('')
        self.status.set('새 학습 준비 · 아래 시작 버튼으로 별도 작업을 만들 수 있습니다.')

    def start_new(self):
        try:
            config = validate(self.configuration(), resolve_seed=True)
            name = datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + uuid.uuid4().hex[:8]
            RESULTS.mkdir(parents=True, exist_ok=True)
            request = REQUESTS / (name + '.json')
            save_state(request, config)
            self.run = RESULTS / name
            self.launch(['--config', str(request), '--output', str(self.run)])
            self.resume.set(str(self.run))
            self.persist()
        except (ValueError, OSError) as error:
            messagebox.showerror('시작 확인', str(error), parent=self.root)

    def start_resume(self):
        try:
            if not self.resume.get():
                raise ValueError('이어 학습할 기록 폴더를 선택하세요.')
            self.run = Path(self.resume.get()).resolve()
            meta = load_state(self.run / 'training.json') or {}
            if meta.get('trainerKind') != KIND:
                raise ValueError('이 학습기의 기록 폴더가 아닙니다.')
            target = int(self.controls['episodes'].get())
            if target <= meta.get('completedEpisodes', 0):
                raise ValueError(f"총 학습 경기 수를 저장된 {meta['completedEpisodes']:,}경기보다 크게 설정하세요.")
            self.launch(['--resume', str(self.run), '--episodes', str(target)])
            self.persist()
        except (ValueError, OSError) as error:
            messagebox.showerror('이어 학습 확인', str(error), parent=self.root)

    def launch(self, args):
        # stdout is outside the run so the engine can enforce an unused folder.
        logs = LOGS
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / (self.run.name + '.log')).open('a', encoding='utf-8') as stream:
            self.process = subprocess.Popen([sys.executable, '-u', '-m', 'unity_bridge.reward_generation.train', *args],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONIOENCODING='utf-8'), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.processes[str(self.run)] = self.process
        self.status.set('학습 모델과 병렬 환경을 준비하고 있습니다.')
        self.refresh_jobs()

    def refresh_jobs(self):
        paths = sorted(RESULTS.glob('*/training.json'), reverse=True)
        self.job_combo.configure(values=[p.parent.name for p in paths])
        if self.run:
            self.job.set(self.run.name)

    def select_job(self, _=None):
        self.run = RESULTS / self.job.get()
        self.resume.set(str(self.run))
        meta = load_state(self.run / 'training.json')
        if meta:
            self.apply_config(meta['config'])
        self.persist()

    def stop(self):
        if self.run and (self.run / 'training.json').exists():
            (self.run / 'stop.request').touch()
            self.status.set('진행 중인 경기 묶음을 정리하고 모델·세대 목록을 저장합니다.')

    def open_models(self):
        folder = model_directory(self.run) if self.run else MODEL_ROOT
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)

    def poll(self):
        self.refresh_model_paths(self.run)
        if self.run:
            p = load_state(self.run / 'progress.json')
            if p:
                total = p['target']
                self.bar['value'] = p['episodes'] / max(total, 1) * 100
                labels = dict(running='학습 중', stopped='중단·저장됨', completed='완료', failed='오류')
                self.status.set(f'{labels.get(p["status"], p["status"])} · {p["episodes"]:,} / {total:,}경기 · '
                    f'승/패/무 {p["wins"]:,}/{p["losses"]:,}/{p["draws"]:,} · '
                    f'전체 경기 기준 승률 {100*p["wins"]/max(p["episodes"],1):.1f}% · CPU {p["workers"]}개')
                opponents = ', '.join(str(e['generation']) for e in p['opponents']) or '무보류'
                self.generation_status.set(f'완료 세대 {p["generation"]} · 현재 상대: {opponents} · 다음 교체 기준 {p["nextGenerationAt"]:,}경기')
            log = LOGS / (self.run.name + '.log')
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 6000))
                    tail = stream.read().decode('utf-8', errors='replace')
                self.log.configure(state='normal')
                self.log.delete('1.0', 'end')
                self.log.insert('end', tail)
                self.log.see('end')
                self.log.configure(state='disabled')
            process = self.processes.get(str(self.run))
            if process and process.poll() not in (None, 0):
                self.status.set('학습 실행 오류 · 아래 로그를 확인하세요. 기존 모델은 덮어쓰지 않습니다.')
        self.poll_job = self.root.after(1000, self.poll)

    def close(self):
        if self.poll_job is not None:
            self.root.after_cancel(self.poll_job)
        if self.save_job is not None:
            self.root.after_cancel(self.save_job)
        self.persist()
        self.root.destroy()


def main():
    root = tk.Tk()
    GenerationTrainer(root)
    root.mainloop()


if __name__ == '__main__':
    main()
