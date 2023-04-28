import fileinput
from os.path import join, dirname

version_file = join(dirname(dirname(__file__)), "ovos_tts_plugin_piper", "version.py")

alpha_var_name = "VERSION_ALPHA"

for line in fileinput.input(version_file, inplace=True):
    if line.startswith(alpha_var_name):
        print(f"{alpha_var_name} = 0")
    else:
        print(line.rstrip('\n'))
