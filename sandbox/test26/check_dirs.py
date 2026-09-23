import os
print("CWD:", os.getcwd())
print("CWD contents:", os.listdir('.'))
print("example/ contents:", os.listdir('example'))
if os.path.exists('example/template'):
    print("example/template/ contents:", os.listdir('example/template'))
