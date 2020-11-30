
import csv
import numpy as np
import matplotlib.pyplot as plt


file_name="perf.txt"
f = open(file_name, 'r')
reader = csv.reader(f)
width = 0
inch = 0
max_p = 0
max_p2 = 0
max_t = 0
max_n = "none"
Dct_x = {}
Dct_y = {}
Dct_p = {}
Dct_t = {}
Dct_x[max_n] = []
Dct_y[max_n] = []
Dct_p[max_n] = []
Dct_t[max_n] = []
for row in reader:
    kernel = int(row[0])
    stride = int(row[1])
    w = int(row[2])
    c = int(row[3])
    num = int(row[4])
    okm = row[5]
    t0 = float(row[6])
    t1 = float(row[7])
    test = float(row[8])
    if(Dct_x.get(okm) == None): 
        Dct_x[okm] = []
        Dct_y[okm] = []
        Dct_p[okm] = []
        Dct_t[okm] = []

    if(width != w or inch != c):
        Dct_x[max_n].append(width)
        Dct_y[max_n].append(inch)
        Dct_p[max_n].append(max_p-max_p2)
        Dct_t[max_n].append(max_t)
        width = w
        inch = c
        max_p2 = 0
        max_p =t0/t1
        max_n = okm
        max_t = test

    else:
        p = t0/t1
        if(p > max_p):
            max_p2 = max_p
            max_p = p
            max_n = okm
            max_t = test


plt.subplot(211)
for key in Dct_x:
    if key == 'none': continue
    area = []
    for v in Dct_p[key]:
        if(v < 3):
            area.append(1)
        elif(3 <= v and v < 6):
            area.append(4)
        elif(6 <= v and v < 9):
            area.append(6)
        elif(9 <= v and v < 12):
            area.append(8)
        else:
            area.append(9)
    #plt.scatter(Dct_x[key], Dct_y[key], marker='.', s=area, label=str(key))
    plt.scatter(Dct_x[key], Dct_y[key], s=area, label=str(key))
plt.legend(loc = 'best', fontsize="xx-small")
plt.xlabel('width(=height)')
plt.ylabel('inch(=outch)')

plt.subplot(212)
for key in Dct_x:
    if key == 'none': continue
    area = []
    for v in Dct_t[key]:
        if(v < 1):
            area.append(1)
        elif(1 <= v and v < 100):
            area.append(5)
        elif(100 <= v and v < 1000):
            area.append(10)
        elif(100 <= v and v < 10000):
            area.append(30)
        else:
            area.append(100)
    #plt.scatter(Dct_x[key], Dct_y[key], marker='.', s=area, label=str(key))
    plt.scatter(Dct_x[key], Dct_y[key], s=area, label=str(key))
plt.xlabel('width(=height)')
plt.ylabel('inch(=outch)')
plt.subplots_adjust(hspace=0)


plt.savefig("perf.png", dpi=1000)
f.close()
