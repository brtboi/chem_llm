from chem_llm.tools import generate_cif
import json

result = generate_cif(
    composition="TiO2",
    output_path="sandbox/test_claude/structures/structure_000_raw.cif",
    spacegroup_symbol="P4_2/mnm",
    spacegroup_number=136,
)
print(json.dumps(result, indent=2))
