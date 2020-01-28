import matplotlib.pyplot as plt
import numpy as np
import pickle


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color='gray')
    plt.setp(bp['means'], color='gray')


folder = 'results/'
C = 6
if C == 6:
    files = ['NCT_s1000.pkl', 'UBT_C6_s1000.pkl', 'DWT_C6_s1000.pkl',
             'RHT1_C6_s1000.pkl', 'RHT3_C6_s1000.pkl', 'UST5_C6_s1000.pkl']
elif C == 10:
    files = ['NCT_s1000.pkl', 'UBT_C10_s1000.pkl', 'DWT_C10_s1000.pkl',
             'RHT1_C10_s1000.pkl', 'RHT3_C10_s1000.pkl', 'UST5_C10_s1000.pkl']


names = ['No\nControl', 'UBT', 'DWT', 'RHT\n'+r'$k=1$', 'RHT\n'+r'$k=3$', 'UST']

data_healthy_trees = []
data_healthy_urban = []

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure()

for f in files:
    pkl_file = open(folder + f, 'rb')
    results = pickle.load(pkl_file)
    pkl_file.close()

    healthy_trees = [100*results[s]['healthy_trees'] for s in results.keys()]
    healthy_urban = [100*results[s]['healthy_urban'] for s in results.keys()]

    data_healthy_trees.append(healthy_trees)
    data_healthy_urban.append(healthy_urban)

boxprops = dict(linewidth=1.5)
whisprops = dict(linewidth=1)
medianprops = dict(linewidth=1.5)
meanprops = dict(linestyle='--', linewidth=1.5, color='gray')
left = plt.boxplot(data_healthy_trees, showmeans=True, meanline=True, showfliers=False, whis='range',
                   boxprops=boxprops, medianprops=medianprops, whiskerprops=whisprops, meanprops=meanprops,
                   positions=np.array(range(len(files)))*3.0-0.6, widths=1)
right = plt.boxplot(data_healthy_urban, showmeans=True, meanline=True, showfliers=False, whis='range',
                    boxprops=boxprops, medianprops=medianprops, whiskerprops=whisprops, meanprops=meanprops,
                    positions=np.array(range(len(files)))*3.0+0.6, widths=1)
set_box_color(left, 'C0')
set_box_color(right, 'orange')

plt.tight_layout(pad=0)

if C == 6:
    plt.plot([], c='C0', label='Trees')
    plt.plot([], c='orange', label='Urban Areas')
    plt.plot([], c='gray', linestyle='--', label='Mean')
    plt.plot([], c='gray', label='Median')
    plt.legend(loc='lower left', ncol=2, fontsize=16, mode='expand', bbox_to_anchor=(0, 1.02, 1, 0.2), borderaxespad=0)


plt.xticks(range(0, len(names)*3, 3), names, fontsize=18)
plt.xlim(-2, len(names)*3-1)

plt.yticks(range(0, 110, 10), fontsize=16)
plt.ylabel('Remaining Healthy (\%)', fontsize=18)

# plt.show()
plt.savefig('results_C' + str(C) + '.pdf', dpi=300, bbox_inches='tight')
