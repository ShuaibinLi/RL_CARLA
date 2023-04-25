import subprocess

count = 0
print('carla start')
while True:
    count += 1
    returncode = subprocess.call(['./CarlaUE4.sh', '-windowed', '-carla-port=2031', '-Renderoffscreen'])
    
    if returncode != 0:
        print(f"Carla simulation interrupted, attempt #{count}")
        continue
