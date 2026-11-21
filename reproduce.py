"""Reproduce completed seed-42 experiments in a fresh checkout."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run([sys.executable, script, *map(str,args)], cwd=ROOT, check=True)


def main():
    results = ROOT/'results'
    if results.exists() and any(results.iterdir()):
        raise SystemExit("results/ already contains files. Use the individual commands in README for an existing run.")
    run('experiment.py', 'all', '--seeds', 42)
    run('summarize_results.py')
    for group, sub in [('F','compact'),('G','lightweight'),('I','serial')]:
        folder = results/sub
        folder.mkdir()
        shutil.copy2(results/'manifest.json',folder/'manifest.json')
        run('experiment.py','train','--out',folder,'--groups',group,'--seeds',42)
    for group in ['B','G']:
        run('cnn_occlusion.py','--group',group)
    run('evaluate_compact.py')
    run('evaluate_compact.py','--serial')
    run('evaluate_compact.py','--lightweight')


if __name__ == '__main__':
    main()
