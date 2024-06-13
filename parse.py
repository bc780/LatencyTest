import re

file = open("slurm-26728462.out", "r")
info = file.read()

sends = re.findall('sent [0-9]+ -> [0-9]+ # [0-9]+ at [0-9.]+', info)
recvs = re.findall('recv [0-9]+ -> [0-9]+ # [0-9]+ at [0-9.]+', info)

sendDict = dict()
recvDict = dict()

for i in sends:
    sender = re.search('(?<=sent )[0-9]+',i).group()
    receiver = re.search('(?<=-> )[0-9]+',i).group()
    multiple = re.search('(?<=# )[0-9]+',i).group()
    time = float(re.search('(?<=at )[.0-9]+',i).group())
    sendDict[(sender,receiver,multiple)] = time
for i in recvs:
    sender = re.search('(?<=recv )[0-9]+',i).group()
    receiver = re.search('(?<=-> )[0-9]+',i).group()
    multiple = re.search('(?<=# )[0-9]+',i).group()
    time = float(re.search('(?<=at )[.0-9]+',i).group())
    recvDict[(sender,receiver,multiple)] = time

diffDict = dict()
for i in sendDict:
    diffDict[i] = recvDict[i] - sendDict[i]
    if(diffDict[i] < 0):
        print(i)
        print("Failed")

print(diffDict)

file.close()