import inspect
import chem_llm.kpath as kpath_mod

print("Functions in chem_llm.kpath:")
for name, obj in inspect.getmembers(kpath_mod):
    if inspect.isfunction(obj):
        sig = inspect.signature(obj)
        print(f"  {name}{sig}")
