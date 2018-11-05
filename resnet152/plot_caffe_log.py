import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#首先要: ./tools/extra/parse_log.sh nohup.out生成nohup.out.train

f = open("nohup.out.train", 'r').readlines()

data = []
for line in f[1:]:
	if 1:
		a = []
		for i in line.split(" "):
			if i != "":
				a.append(i)
		
		temp = (int(a[0]), float(a[-2]))
		print temp
		data.append(temp)

y = np.array(data)

plt.figure(figsize=(7,5))
plt.ylim(0,5)

plt.plot(y[:,0], y[:,1])
plt.xlabel('iteration')
plt.ylabel('train loss')
plt.show()
