import tagStruct
import matplotlib.pyplot as plt
import math
import numpy as np

filePath = "data/Robbie2018Tag.csv"
tag = tagStruct.Tag(filePath)
colNames = ['Time', 'Depth', 'Temperature', 'Light Level']
tag.readcsv(colNames)
tag.getDives(30)



lengths = [len(tag.dives[i]) for i in range(0, len(tag.dives))]
buckets = np.linspace(math.ceil(min(lengths)), math.floor(max(lengths)), 30)
plt.hist(lengths, bins = buckets)
plt.title("Distribution of dive lengths in time")
plt.xlabel("Number of measurements taken during dive")
plt.ylabel("Frequency")
#plt.show()


tag.addVelocityColumn()
counts = tag.createCounts()
