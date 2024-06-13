import re

file = open("slurm-26734351.out", "r")
info = file.read()

sends = re.findall('sent [0-9]+ -> [0-9]+ # [0-9]+ at [0-9.]+', info)
recvs = re.findall('recv [0-9]+ -> [0-9]+ # [0-9]+ at [0-9.]+', info)

sendDict = dict()
recvDict = dict()

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
for i in sendDict:
    diffDict[i] = recvDict[i] - sendDict[i]
    if(diffDict[i] < 0):
        print(i)
        print("Failed")

print(diffDict)
print(diffDict[(0,1,0)])
print(diffDict[(0,1,1)])
print(diffDict[(0,1,2)])
print(diffDict[(0,1,3)])
print(diffDict[(0,1,4)])
print(diffDict[(0,1,5)])

file.close()