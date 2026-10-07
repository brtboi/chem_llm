import re

# Read pw.out and find the 'number of electrons' line
with open('calculations/000/pw.out') as f:
    content = f.read()

# Find the number of electrons
match = re.search(r'number of electrons\s*[:=]\s*([\d.]+)', content)
if match:
    n_electrons = float(match.group(1))
    print(f"Number of electrons from pw.out: {n_electrons}")
    occupied_bands = n_electrons / 2  # spin-unpolarized
    print(f"Occupied bands (spin-unpolarized): {occupied_bands}")
    
    # Check against nbnd=40
    nbnd = 40
    if occupied_bands <= nbnd:
        print(f"OK: occupied bands ({occupied_bands}) <= nbnd ({nbnd})")
    else:
        print(f"ERROR: occupied bands ({occupied_bands}) > nbnd ({nbnd}) -- need to increase nbnd")
else:
    # Try alternative pattern
    match = re.search(r'number of electrons\s+([\d.]+)', content)
    if match:
        n_electrons = float(match.group(1))
        print(f"Number of electrons from pw.out: {n_electrons}")
        occupied_bands = n_electrons / 2
        print(f"Occupied bands (spin-unpolarized): {occupied_bands}")
        nbnd = 40
        if occupied_bands <= nbnd:
            print(f"OK: occupied bands ({occupied_bands}) <= nbnd ({nbnd})")
        else:
            print(f"ERROR: occupied bands ({occupied_bands}) > nbnd ({nbnd}) -- need to increase nbnd")
    else:
        print("Could not find 'number of electrons' in pw.out")
        # Print lines containing 'electron' for debugging
        for line in content.split('\n'):
            if 'electron' in line.lower():
                print(f"  {line}")
