# run_snr.py
from sca_framework.snr_analysis import demonstrate_snr_analysis

demonstrate_snr_analysis(
    traces_path="src/py/data/traces_1_filtered.npz",
    correct_key=0x10a58869d74be5a374cf867cfb473859,
    byte_index=0
)
