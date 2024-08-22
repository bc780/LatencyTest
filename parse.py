import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# outFiles = ["a.out","b.out","c.out","d.out","e.out"]
# outFiles = ["slurm-29560476.out"]
# outFiles = ["output_1.out"]
outFiles = []
for i in range(1,4):
    outFiles.append("output_" + str(i) + ".out")

bandwidthData = dict()
nodeData = dict()
allData = dict()

for f in outFiles:
    file = open(f, "r")
    info = file.read()

    sends = re.findall('sent [0-9]+ -> [0-9]+ # [0-9]+ at [0-9.]+', info)
    recvs = re.findall('recv [0-9]+ -> [0-9]+ # [0-9]+ at [0-9.]+', info)
    checksums = re.findall('CHECKSUM [0-9]+ [0-9]+ [0-9]+: [0-9.]+', info)
    nodeReduce = re.findall('Node Reduce rank [0-9]+ at [0-9.]+', info)
    allReduce = re.findall('All Reduce rank [0-9]+ at [0-9.]+', info)


    sendDict = dict()
    recvDict = dict()
    nodeDict = dict()
    allDict = dict()


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
    for i in nodeReduce:
        rank = int(re.search('(?<=rank )[0-9]+',i).group())
        time = int(re.search('(?<=at )[0-9.]+',i).group())
        nodeDict[rank] = time
    for i in allReduce:
        rank = int(re.search('(?<=rank )[0-9]+',i).group())
        time = int(re.search('(?<=at )[0-9.]+',i).group())
        allDict[rank] = time


    diffDict = dict()
    count = 0
    for i in sendDict:
        diffDict[i] = recvDict[i]
        # if(diffDict[i] < 0):
        #     print(i)
        #     print(f)
        #     print("Failed")
        #     count += 1


    for i in diffDict:
        if i in bandwidthData:
            bandwidthData[i].append(diffDict[i])
        else:
            bandwidthData[i] = []
            bandwidthData[i].append(diffDict[i])
    for i in nodeDict:
        if i in nodeData:
            nodeData[i].append(nodeDict[i])
        else:
            nodeData[i] = []
            nodeData[i].append(nodeDict[i])
    for i in allDict:
        if i in allData:
            allData[i].append(allDict[i])
        else:
            allData[i] = []
            allData[i].append(allDict[i])

    
    

def nsToGbs(time):
    return format(8000000000.0/float(time),'.4f') 

for i in range(4):
    nodeReduceBandwidths = [0]*16
    for j in range(4):
        nodeReduceBandwidths[j] = np.average(nodeData[i*4+j])
    averageNodeReduce = nsToGbs(np.average(nodeReduceBandwidths))
    print("Node", i, "Bandwidth: ", averageNodeReduce)
    

allReduceBandwidths = [0]*16
for i in range(16):
    allReduceBandwidths[i] = np.average(allData[i])
averageAllReduce = nsToGbs(np.average(allReduceBandwidths))
print("All Reduce Bandwidth: ", averageAllReduce)

bandwidths = np.zeros((16,16),dtype=float)
sd = np.zeros((16,16),dtype=float)
for i in bandwidthData:
    bandwidths[i[0]][i[1]] = nsToGbs(np.average(bandwidthData[i]))
    sd[i[0]][i[1]] = nsToGbs(np.std(bandwidthData[i]))
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
red_green = [(1,0,0),(0,1,0)]
map = colors.LinearSegmentedColormap.from_list('red_green',red_green)

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

fig, ax = plt.subplots(figsize=(18,18))
ax.axis("tight")
ax.axis("off")

table = ax.table(cellText=sd,rowLabels=rows,colLabels=columns,loc='center')

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