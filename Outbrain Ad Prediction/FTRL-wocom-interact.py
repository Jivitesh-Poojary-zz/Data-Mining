# -*- coding: utf-8 -*-
#!/N/soft/rhel6/python/2.7.3/bin/python
"""
Thanks to tinrtgu for the wonderful base script
Use pypy for faster computations.!
"""
import csv
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt

import time
start_time = time.time()


data_path = "./"
train = data_path+'clicks_train.csv'               # path to training file
test = data_path+'clicks_test.csv'                 # path to testing file
submission = 'sub_proba.csv'  # path of to be outputted submission file

# B, model
alpha = .1  # learning rate
beta = 0.   # smoothing parameter for adaptive learning rate
L1 = 0.    # L1 regularization, larger value means more regularized
L2 = 0.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 20             # number of weights to use
interaction = True      # whether to enable poly2 feature interactions

# D, training/validation
epoch = 1       # learn training data for N passes
holdafter = None   # data after date N (exclusive) are used as validation
holdout = None  # use every N training instance for holdout validation

class ftrl_proximal(object):
   
    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2
        self.D = D
        self.interaction = interaction
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):
        
        yield 0

        for index in x:
            yield index

        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
       
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        n = self.n
        z = self.z
        w = {}

        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            if sign * z[i] <= L1:
                w[i] = 0.
            else:
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]
        self.w = w
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        alpha = self.alpha
        n = self.n
        z = self.z
        w = self.w
        g = p - y
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(path, D):
  
    for t, row in enumerate(DictReader(open(path))):
        # process id
        disp_id = int(row['display_id'])
        ad_id = int(row['ad_id'])

        # process clicks
        y = 0.
        if 'clicked' in row:
            if row['clicked'] == '1':
                y = 1.
            del row['clicked']

        x = []
        for key in row:
            x.append(abs(hash(key + '_' + row[key])) % D)

        row = prcont_dict.get(ad_id, [])		
        # build x
        ad_doc_id = -1
        for ind, val in enumerate(row):
            if ind==0:
                ad_doc_id = int(val)
            x.append(abs(hash(prcont_header[ind] + '_' + val)) % D)

        row = event_dict.get(disp_id, [])
        ## build x
        disp_doc_id = -1
        for ind, val in enumerate(row):
            if ind==0:
                uuid_val = val
            if ind==1:
                disp_doc_id = int(val)
            x.append(abs(hash(event_header[ind] + '_' + val)) % D)			

        if (ad_doc_id in leak_uuid_dict) and (uuid_val in leak_uuid_dict[ad_doc_id]):
            print("1--- %s seconds ---" % (time.time() - start_time))
            x.append(abs(hash('leakage_row_found_1'))%D)
        else:
            x.append(abs(hash('leakage_row_not_found'))%D)
            
        yield t, disp_id, ad_id, x, y

start = datetime.now()

learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

print("Content..")
with open(data_path + "promoted_content.csv") as infile:
	prcont = csv.reader(infile)
	#prcont_header = (prcont.next())[1:]
	prcont_header = next(prcont)[1:]
	prcont_dict = {}
	for ind,row in enumerate(prcont):
		prcont_dict[int(row[0])] = row[1:]
del prcont

print("Events..")
with open(data_path + "events.csv") as infile:
	events = csv.reader(infile)
	#events.next()
	next(events)
	event_header = ['uuid', 'document_id', 'platform', 'geo_location', 'loc_country', 'loc_state', 'loc_dma']
	event_dict = {}
	for ind,row in enumerate(events):
		tlist = row[1:3] + row[4:6]
		loc = row[5].split('>')
		if len(loc) == 3:
			tlist.extend(loc[:])
		elif len(loc) == 2:
			tlist.extend( loc[:]+[''])
		elif len(loc) == 1:
			tlist.extend( loc[:]+['',''])
		else:
			tlist.append(['','',''])	
		event_dict[int(row[0])] = tlist[:] 
	print(len(event_dict))
del events

print("Leakage file..")
leak_uuid_dict= {}
for e in range(epoch):
    loss = 0.
    count = 0
    date = 0

    for t, disp_id, ad_id, x, y in data(train, D):  # data is a generator
        p = learner.predict(x)

        if (holdafter and date > holdafter) or (holdout and t % holdout == 0):
            loss += logloss(p, y)
            count += 1
        else:
            learner.update(x, p, y)

print("--- %s seconds ---" % (time.time() - start_time))			
       
with open(data_path + submission, 'w') as outfile:
    outfile.write('display_id,ad_id,clicked\n')
    print('Testing 123')
    for t, disp_id, ad_id, x, y in data(test, D):
        p = learner.predict(x)
        outfile.write('%s,%s,%s\n' % (disp_id, ad_id, str(p)))
