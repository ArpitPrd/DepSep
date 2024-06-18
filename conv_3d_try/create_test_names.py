import os

files = os.listdir('Test_Clean')
print(len(files))
f = open('test_names.txt', 'w')
for line in files:
    f.write(line)

f.close()