from chem_llm.tools import get_pseudopotential
import json

for el in ["Ti", "O"]:
    result = get_pseudopotential(
        element=el,
        output_path=f"sandbox/test_claude/template/{el}.upf",
        kind="nc",
        relativity="sr",
        generator="pbesol",
        accuracy="stringent",
        format="upf",
        hint_level="normal",
    )
    print(el, json.dumps(result, indent=2))
