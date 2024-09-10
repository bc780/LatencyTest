import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

world_size = 16
iterations = 2

# outFiles = ["a.out","b.out","c.out","d.out","e.out"]
# outFiles = ["slurm-29560476.out"]
# outFiles = ["output_1.out"]
outFiles = []
for i in range(1,iterations+1):
    outFiles.append("output_" + str(i) + ".out")

bandwidthData = dict()
nodeData = dict()
allData = dict()
revData = dict()

for f in outFiles:
    file = open(f, "r")
    info = file.read()

    sends = re.findall('sent [0-9]+ -> [0-9]+ # [0-9]+ at [0-9.]+', info)
    recvs = re.findall('recv [0-9]+ -> [0-9]+ # [0-9]+ at [0-9.]+', info)
    checksums = re.findall('CHECKSUM [0-9]+ [0-9]+ [0-9]+: [0-9.]+', info)
    nodeReduce = re.findall('Node Reduce rank [0-9]+ at [0-9.]+', info)
    allReduce = re.findall('All Reduce rank [0-9]+ at [0-9.]+', info)
    reverseRecvs = re.findall('recv reverse [0-9]+ -> [0-9]+ # [0-9]+ at [0-9.]+', info)


    sendDict = dict()
    recvDict = dict()
    nodeDict = dict()
    allDict = dict()
    revDict = dict()


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
    for i in reverseRecvs:
        sender = int(re.search('(?<=recv )[0-9]+',i).group())
        receiver = int(re.search('(?<=-> )[0-9]+',i).group())
        multiple = int(re.search('(?<=# )[0-9]+',i).group())
        time = int(re.search('(?<=at )[.0-9]+',i).group())
        revDict[(sender,receiver,multiple)] = time



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
    for i in revDict:
        if i in revData:
            revData[i].append(revDict[i])
        else:
            revData[i] = []
            revData[i].append(revDict[i])

    
    
#converts time into gigabytes
def nsToGBs(time):
    return format(8000000000.0/float(time),'.2f') 

#calculates and prints node all reduce
for i in range(4):
    nodeReduceBandwidths = [0]*world_size
    for j in range(4):
        nodeReduceBandwidths[j] = np.average(nodeData[i*4+j])
    averageNodeReduce = nsToGBs(np.average(nodeReduceBandwidths))
    print("Node", i, "Bandwidth: ", averageNodeReduce)
    
#calculates and prints all all reduce
allReduceBandwidths = [0]*world_size
for i in range(world_size):
    allReduceBandwidths[i] = np.average(allData[i])
averageAllReduce = nsToGBs(np.average(allReduceBandwidths))
print("All Reduce Bandwidth: ", averageAllReduce)



#formatting bandwidth and stddev data into a matrix
bandwidths = np.zeros((world_size,world_size),dtype=float)
sd = np.zeros((world_size,world_size),dtype=float)
for i in bandwidthData:
    bandwidths[i[0]][i[1]] = nsToGBs(np.average(bandwidthData[i]))
    temp = dict()
    temp[i] = []
    for j in bandwidthData[i]:
        temp[i].append(float(nsToGBs(j)))
    sd[i[0]][i[1]] = format(np.std(temp[i]),'.2f')


#formatting reverse data into a matrix
reverseBands = np.zeros((world_size,world_size),dtype=float)
reverseSd = np.zeros((world_size,world_size),dtype=float)
for i in revData:
    reverseBands[i[0]][i[1]] = nsToGBs(np.average(revData[i]))
    temp = dict()
    temp[i] = []
    for j in revData[i]:
        temp[i].append(float(nsToGBs(j)))
    reverseSd[i[0]][i[1]] = format(np.std(temp[i]),'.2f')

#column and row labels
columns = [str(x) for x in range(bandwidths.shape[0])]
rows = [str(x) for x in range(bandwidths.shape[1])]

#plt settings for bandwidth table
fig, ax = plt.subplots(figsize=(10,10))
ax.axis("tight")
ax.axis("off")
table = ax.table(cellText=bandwidths,rowLabels=rows,colLabels=columns,loc='center',cellLoc='center')
table.scale(1,2)
table.auto_set_font_size(False)
table.set_fontsize(10)
fig.subplots_adjust(left=0.1, top=1)
plt.text(0.5, 0.95, 'Bandwidth', ha='center', va='top', transform=ax.transAxes, fontsize=20, weight='bold',fontname='serif')


#normalizing for colors
non_zero_data = bandwidths[bandwidths != 0]
min_val = non_zero_data.min()
max_val = non_zero_data.max()
norm = colors.Normalize(vmin=min_val,vmax=max_val)
red_green = [(1,0,0),(0,1,0)]
map = colors.LinearSegmentedColormap.from_list('red_green',red_green)

#cell settings for bandwidth table
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

#plt settings for stddev table
fig, ax = plt.subplots(figsize=(10,10))
ax.axis("tight")
ax.axis("off")
table = ax.table(cellText=sd,rowLabels=rows,colLabels=columns,loc='center')
table.scale(1,2)
table.auto_set_font_size(False)
table.set_fontsize(10)
fig.subplots_adjust(left=0.1, top=1)
plt.text(0.5, 0.95, 'Standard Deviations', ha='center', va='top', transform=ax.transAxes, fontsize=20, weight='bold',fontname='serif')


#normalizing for colors
non_zero_data = sd[sd != 0]
min_val = non_zero_data.min()
max_val = non_zero_data.max()
norm = colors.Normalize(vmin=min_val,vmax=max_val)
green_red = [(0,1,0),(1,0,0)]
map = colors.LinearSegmentedColormap.from_list('green_red',green_red)

#cell settings for stddev table
for i in range(bandwidths.shape[0]):
    for j in range(bandwidths.shape[1]):
        cell_value = sd[i, j]
        if cell_value == 0:
            cell_color = (0, 0, 0, 1)  # black for zero entries
        else:
            cell_color = map(norm(cell_value))  # Normalize and map to color
        cell = table[(i+1, j)]  # (i+1, j) because the first row is the header
        cell.set_facecolor(cell_color)
        cell.set_text_props(color='black' if cell_value == 0 else 'white')  # Set text color for better contrast

#plt settings for reverse bandwidth table
fig, ax = plt.subplots(figsize=(10,10))
ax.axis("tight")
ax.axis("off")
table = ax.table(cellText=reverseBands,rowLabels=rows,colLabels=columns,loc='center',cellLoc='center')
table.scale(1,2)
table.auto_set_font_size(False)
table.set_fontsize(10)
fig.subplots_adjust(left=0.1, top=1)
plt.text(0.5, 0.95, 'Reverse Bandwidth', ha='center', va='top', transform=ax.transAxes, fontsize=20, weight='bold',fontname='serif')


#normalizing for colors
non_zero_data = reverseBands[reverseBands != 0]
min_val = non_zero_data.min()
max_val = non_zero_data.max()
norm = colors.Normalize(vmin=min_val,vmax=max_val)
red_green = [(1,0,0),(0,1,0)]
map = colors.LinearSegmentedColormap.from_list('red_green',red_green)

#cell settings for reverse bandwidth table
for i in range(bandwidths.shape[0]):
    for j in range(bandwidths.shape[1]):
        cell_value = reverseBands[i, j]
        if cell_value == 0:
            cell_color = (0, 0, 0, 1)  # black for zero entries
        else:
            cell_color = map(norm(cell_value))  # Normalize and map to color
        cell = table[(i+1, j)]  # (i+1, j) because the first row is the header
        cell.set_facecolor(cell_color)
        cell.set_text_props(color='black' if cell_value == 0 else 'white')  # Set text color for better contrast

#plt settings for reverse standard deviation table
fig, ax = plt.subplots(figsize=(10,10))
ax.axis("tight")
ax.axis("off")
table = ax.table(cellText=reverseSd,rowLabels=rows,colLabels=columns,loc='center',cellLoc='center')
table.scale(1,2)
table.auto_set_font_size(False)
table.set_fontsize(10)
fig.subplots_adjust(left=0.1, top=1)
plt.text(0.5, 0.95, 'Reverse Standard Deviation', ha='center', va='top', transform=ax.transAxes, fontsize=20, weight='bold',fontname='serif')


#normalizing for colors
non_zero_data = reverseSd[reverseSd != 0]
min_val = non_zero_data.min()
max_val = non_zero_data.max()
norm = colors.Normalize(vmin=min_val,vmax=max_val)
red_green = [(1,0,0),(0,1,0)]
map = colors.LinearSegmentedColormap.from_list('red_green',red_green)

#cell settings for reverse standard deviation table
for i in range(bandwidths.shape[0]):
    for j in range(bandwidths.shape[1]):
        cell_value = reverseSd[i, j]
        if cell_value == 0:
            cell_color = (0, 0, 0, 1)  # black for zero entries
        else:
            cell_color = map(norm(cell_value))  # Normalize and map to color
        cell = table[(i+1, j)]  # (i+1, j) because the first row is the header
        cell.set_facecolor(cell_color)
        cell.set_text_props(color='black' if cell_value == 0 else 'white')  # Set text color for better contrast

plt.show()

file.close()