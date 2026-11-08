"""Exercise persistent desktop settings and shared attack/alpha values without training."""
import tempfile
from pathlib import Path
import tkinter as tk
from unity_bridge.generation_trainer import GenerationTrainer
from unity_bridge.generation_config import CUSTOM
from unity_bridge.storage import ROOT
from unity_bridge.trainer_state import save_state


def main():
    errors = []
    with tempfile.TemporaryDirectory(prefix='generation-ui-', dir=ROOT / 'storage-reports') as temp:
        path = Path(temp) / 'state.json'
        root = tk.Tk()
        root.withdraw()
        root.report_callback_exception = lambda *args: errors.append(str(args))
        ui = GenerationTrainer(root, state_path=path)
        ui.select_group('A', 'professional')
        ui.select_group('B', 'amateur')
        ui.controls['generationGames'].set('1234')
        ui.controls['retainedGenerations'].set('2')
        ui.controls['seed'].set('')
        ui.controls['episodes'].set('43210')
        ui.custom['A'].set(True)
        ui.reward_vars['A']['mode'].set('범위')
        ui.reward_vars['A']['min'].set('.2')
        ui.reward_vars['A']['max'].set('.8')
        ui.reward_vars['A']['step'].set('.1')
        ui.inputs['cooldown']['value'].set('500')
        root.update_idletasks()
        before = ui.configuration()
        assert CUSTOM in before['presetChoices']['A']
        assert len(before['presetChoices']['A']) == 11 and len(before['presetChoices']['B']) == 10
        assert before['settings']['A.cooldown']['value'] == before['settings']['B.cooldown']['value'] == 500
        assert '0.4' in ui.reward_previews['A'].get()
        ui.close()
        root = tk.Tk()
        root.withdraw()
        root.report_callback_exception = lambda *args: errors.append(str(args))
        restored = GenerationTrainer(root, state_path=path)
        after = restored.configuration()
        assert before == after
        # Interrupted numeric edits also survive restart without blocking saving.
        restored.controls['generationGames'].set('')
        restored.close()
        root = tk.Tk()
        root.withdraw()
        restored = GenerationTrainer(root, state_path=path)
        assert restored.controls['generationGames'].get() == ''
        restored.close()
        assert not errors, errors
        save_state(ROOT / 'storage-reports/generational-ui-verification.json', dict(passed=True,
            professionalRaw=10, amateurRaw=10, presetsAndCustom=True, sharedCooldown=500,
            alphaPreview=True, blankSeed=True, settingsPersisted=True, partialEditPersisted=True))
        print('PASS: preset choices, shared attack, alpha preview and all settings survive restart')


if __name__ == '__main__':
    main()
