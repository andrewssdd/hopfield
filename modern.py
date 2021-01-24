# random stored pattern
# Add noise to first stored pattern and let Hopfield to converge to steady state
# Measure average error.

from models import ModernHopfield, addNoise
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--N', type =int, default=100, help='Number of neurons in Hopfield net')
parser.add_argument('--n-repeats', type =int, default=10, help='Number of repeats for each number of memory runs')
parser.add_argument('--noise-prob', type =float, default=0.1, help='Noise probability for corrupting the query pattern')
parser.add_argument('--max-n-store', type =int, default=20, help='Max number of stored patterns')

opt = parser.parse_args()
print(opt)

N = opt.N
n_repeats = opt.n_repeats
noise_prob = opt.noise_prob
max_n_store = opt.max_n_store
min_diff = {}
for n_store in range(1, max_n_store+1):
    min_diff[n_store] = []
    for r in range(n_repeats):
        X_store = np.random.choice([-1, 1], size=[n_store, N])
        x_query = X_store[0].copy()
        x_query = addNoise(x_query, prob=noise_prob)
        h = ModernHopfield(X_store)
        h.set(x_query)
        n=h.update()
        diff = np.abs((X_store - h.neurons)).sum(1) / 2
        min_diff[n_store].append(np.min(diff))

print('n_memories, average_error_rate')
for n_store in range(1, max_n_store+1):
    print('%d, %1.3f' % (n_store, np.mean(min_diff[n_store]) / N))


