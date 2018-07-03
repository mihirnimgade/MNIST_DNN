import numpy as np

l = np.array([[0.,255.,10.,106.]])
l = np.interp(l, (l.min(), l.max()), (1, 0))

print(l)


