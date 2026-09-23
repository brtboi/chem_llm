import inspect
from ase.calculators.espresso import Espresso, EspressoProfile

# Inspect write_inputfiles signature
print("write_inputfiles signature:")
print(inspect.signature(Espresso.write_inputfiles))
print()

# Inspect the source of write_inputfiles
print("write_inputfiles source:")
print(inspect.getsource(Espresso.write_inputfiles))
print()

# Check if there's a run method
print("Methods of Espresso:")
for name, method in inspect.getmembers(Espresso, predicate=inspect.isfunction):
    if not name.startswith('_'):
        print(f"  {name}{inspect.signature(method)}")
