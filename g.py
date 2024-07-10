import matplotlib.pyplot as plt
from random import randint
import re
import os

rd = open("data.txt", "r").read().split("\n")
rs = open("ans.txt", "r").read().split("\n")
calcs_ls = []
ans_ls = []
counter = 0
for i in rd:
    calcs_ls.append(i.split("_"))
    for j in range(2):
        try:
            calcs_ls[counter][j-1] = int(calcs_ls[counter][j-1])
        except:
            pass
    try:
        ans_ls.append(int(rs[counter]))
    except:
        pass
    counter += 1
calcs_ls.pop(len(calcs_ls)-1)

plt.plot([0, 1], calcs_ls[randint(0, 999)])
print(list(os.listdir))
plt.savefig("graph01.png")