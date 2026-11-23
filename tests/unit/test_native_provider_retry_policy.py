"""SDK transport retries must not silently exceed a scientific attempt budget."""
import ast
from pathlib import Path


def test_all_direct_openai_provider_clients_explicitly_disable_retries():
    root=Path(__file__).resolve().parents[2]/'workers/official_arenas/providers'
    found=0
    for path in root.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='OpenAI':
                found+=1
                policy=next((k.value for k in node.keywords if k.arg=='max_retries'),None)
                assert isinstance(policy,ast.Constant) and policy.value==0, str(path)
    assert found==4
