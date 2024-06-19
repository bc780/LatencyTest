import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

file = open("slurm-26960962.out", "r")
info = file.read()

sends = re.findall('sent [0-9]+ -> [0-9]+ # [0-9]+ at [0-9.]+', info)
recvs = re.findall('recv [0-9]+ -> [0-9]+ # [0-9]+ at [0-9.]+', info)
checksums = re.findall('CHECKSUM [0-9]+ [0-9]+ [0-9]+: [0-9.]+', info)


sendDict = dict()
recvDict = dict()

#validate checksums


for i in sends:
    sender = int(re.search('(?<=sent )[0-9]+',i).group())
    receiver = int(re.search('(?<=-> )[0-9]+',i).group())
    multiple = int(re.search('(?<=# )[0-9]+',i).group())
    time = int(re.search('(?<=at )[.0-9]+',i).group())
    sendDict[(sender,receiver,multiple)] = time
for i in recvs:
    sender = int(re.search('(?<=recv )[0-9]+',i).group())
    receiver = int(re.search('(?<=-> )[0-9]+',i).group())
    multiple = int(re.search('(?<=# )[0-9]+',i).group())
    time = int(re.search('(?<=at )[.0-9]+',i).group())
    recvDict[(sender,receiver,multiple)] = time

diffDict = dict()
count = 0
for i in sendDict:
    diffDict[i] = recvDict[i] - sendDict[i]
    if(diffDict[i] < 0):
        print(i)
        print("Failed")
        count += 1

print(diffDict)

print(count)

bandwidths = np.zeros((16,16),dtype=int)
print(bandwidths)
for i in diffDict:
    print(i[0])
    print(i[1])
    bandwidths[i[0]][i[1]] = diffDict[i]
    print(bandwidths[i[0]][i[1]])
    print(diffDict[i])
print(bandwidths[0][0])

columns = [str(x) for x in range(bandwidths.shape[0])]
rows = [str(x) for x in range(bandwidths.shape[1])]

fig, ax = plt.subplots(figsize=(18,18))
ax.axis("tight")
ax.axis("off")

table = ax.table(cellText=bandwidths,rowLabels=rows,colLabels=columns,loc='center')

non_zero_data = bandwidths[bandwidths != 0]
min_val = non_zero_data.min()
max_val = non_zero_data.max()

norm = colors.Normalize(vmin=min_val,vmax=max_val)
green_red = [(0,1,0),(1,0,0)]
map = colors.LinearSegmentedColormap.from_list('green_red',green_red)

for i in range(bandwidths.shape[0]):
    for j in range(bandwidths.shape[1]):
        cell_value = bandwidths[i, j]
        if cell_value == 0:
            cell_color = (0, 0, 0, 1)  # black for zero entries
        else:
            cell_color = map(norm(cell_value))  # Normalize and map to color
        cell = table[(i+1, j)]  # (i+1, j) because the first row is the header
        cell.set_facecolor(cell_color)
        cell.set_text_props(color='black' if cell_value == 0 else 'white')  # Set text color for better contrast

    
table.scale(1,3)
table.auto_set_font_size(False)
table.set_fontsize(10)
fig.subplots_adjust(left=0.1, top=1)

plt.show()

file.close()