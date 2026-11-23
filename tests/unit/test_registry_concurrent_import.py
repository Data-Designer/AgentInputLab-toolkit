"""Cold imports match the real parallel first-request compatibility failure."""
import subprocess
import sys


def test_concurrent_first_requests_resolve_sibling_profiles():
    script = '''
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from agentinputlab.api import create_intervention
releases = ['plan.plan_and_solve.ps_prompt',
            'plan.plan_and_budget.native_regenerated_soft_budget'] * 4
barrier = Barrier(len(releases))
def one(release):
    barrier.wait()
    instance = create_intervention(release)
    assert instance.release_id == release
with ThreadPoolExecutor(len(releases)) as pool:
    list(pool.map(one, releases))
'''
    subprocess.run([sys.executable, '-c', script],check=True,timeout=30)
