"""Keep this trainer inside the existing application storage categories."""
from unity_bridge.storage import ROOT, CRITICAL_MODEL_ROOT as MODELS

RESULTS = ROOT / 'generational-results' / 'damage-critical'
LOGS = ROOT / 'generational-logs' / 'damage-critical'
PREFERENCES = ROOT / 'app-settings' / 'generational-trainer-damage-critical.json'
REQUESTS = ROOT / 'app-settings' / 'generational-requests' / 'damage-critical'
