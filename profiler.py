import pstats
from code import interact

p = pstats.Stats('profile2.txt')
p.strip_dirs().sort_stats('time').print_stats(10)

interact(local=locals())

