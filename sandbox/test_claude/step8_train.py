from chem_llm.tools.deepseudopot import train_deepseudopot
import json

result = train_deepseudopot(
    inputs_folder="sandbox/test_claude/nn_inputs",
    results_folder="sandbox/test_claude/dpp_results",
    timeout_seconds=1200,
)
print(json.dumps({k: v for k, v in result.items() if k not in ("stdout_tail", "stderr_tail")}, indent=2))
print("--- stdout_tail ---")
print(result.get("stdout_tail", "")[-3000:])
print("--- stderr_tail ---")
print(result.get("stderr_tail", "")[-3000:])
