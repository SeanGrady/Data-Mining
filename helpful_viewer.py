import matplotlib.pyplot as plt
from code import interact
from collections import defaultdict
from ast import literal_eval

def line_generator(fname):
    for line in open(fname):
        yield literal_eval(line)

counts = defaultdict(int)
for line in line_generator('train.json'):
    counts[line['helpful']['outOf']] += 1

x = []
y = []
for outof, count in counts.iteritems():
    x.append(outof)
    y.append(count)

plt.plot(x, y)
plt.show()

interact(local=locals())
