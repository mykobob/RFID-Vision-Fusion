
# coding: utf-8

# In[1]:


import csv
from collections import defaultdict
import math
import pprint as pp
import numpy as np
import os


# In[13]:


def rfid_info(filename):
    rfids = defaultdict(list)
    with open(filename) as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0

        rfid_start = 1 << 64
        rfid_end = 0
        for row in csv_reader:
            epc = row[0] # electronic product code
            peakRssiInDbm = float(row[1]) # Received Signal Strength Indicator
            phase_diff1 = float(row[2]) # Phase difference between [WHAT]
            rfDopplerFreq = row[3] # 
            channelMhz = float(row[4]) # Which channel it was on
            time_millis = float(row[5]) / 1000
            rfid_start = min(rfid_start, float(time_millis))
            rfid_end = max(rfid_end, float(time_millis))
            data = {'peak_rssi': peakRssiInDbm, 'phase_diff': phase_diff1, 'rfDopplerFreq': rfDopplerFreq, 'channelMhz': channelMhz, 'time_millis': time_millis}
            rfids[epc].append(data)
    return rfids

def select_useful_rfid(rfid_info):
    return rfid_info['E20038EBB5953DC9B7693E9C']

def rfid_avgs(rfids):
    avg_rfids_rev = {k: sum([x['peak_rssi'] for x in v]) / len(v) for k, v in rfids.items()}
    return avg_rfids_rev

def wavelength(freq_mhz):
    return 3 * (10 ** 2) / freq_mhz
        
def rfid_dists(measurements):
    epc_dists = {}
    dists = []
    dist = 0 # might need to change to initial phase
    for measurement in measurements:
        wave_len = wavelength(measurement['channelMhz'])
        dists.append(dist)
        dist += measurement['phase_diff'] % (2 * math.pi) * wave_len
        epc_dists[epc] = dists
    
    return epc_dists

def img_metadata(filename):
    timestamps = [float(time) for time in open(filename, 'r').readlines()]
    return timestamps

# Array is sorted
def idx_find(array, target, start=0):
    if start >= len(array):
        return -1
    for idx, val in enumerate(array):
        if idx >= start:
            if val >= target:
                return idx - 1
    return -1

def sanity_check(img_times, tag_times, img2tag):
    diffs = np.zeros((len(img_times), len(tag_times)))
    for img_idx, img_time in enumerate(img_times):
        for tag_idx, tag_time in enumerate(tag_times):
            diffs[img_idx, tag_idx] = abs(tag_time - img_time)
    
    wrong = False
    min_diffs = np.min(diffs, axis=1)
    for img_idx, img_time in enumerate(img_times):
        smallest_tag_idx = np.argmin(diffs[img_idx, :])
        smallest_tag = tag_times[smallest_tag_idx]
        if smallest_tag not in img2tag[img_time]:
            print('img_time:', img_time, 'is wrong.', 'Should be {:.3f}, but got {}'.format(smallest_tag, img2tag[img_time]))
            wrong = True
    
    if not wrong:
        print('All good!')


# In[14]:


def bound_data(tag_times, img_times):
    if tag_times[0] < img_times[0]:
        tag_start = idx_find(tag_times, img_times[0])
        img_start = 0
    else:
        tag_start = 0
        img_start = idx_find(img_times, tag_times[0])
        
    if tag_times[-1] < img_times[-1]:
        tag_end = len(tag_times) - 1
        img_end = idx_find(img_times, tag_times[-1])
    else:
        tag_end = idx_find(tag_times, img_times[-1])
        img_end = len(img_times) - 1
    
    return tag_start, tag_end, img_start, img_end
    

def bucketize(bucket_size, tag_times, img_times):
    tag_start, tag_end, img_start, img_end = bound_data(tag_times, img_times)
    
    all_buckets_start = min(tag_times[tag_start], img_times[img_start]) - bucket_size / 2
    all_buckets_end = max(tag_times[tag_end], img_times[img_end]) + bucket_size / 2
    num_buckets = math.ceil((all_buckets_end - all_buckets_start) / bucket_size)
    bucket_start_times = [all_buckets_start + bucket_size * i for i in range(0, num_buckets)]
    buckets = [([], []) for i in range(len(bucket_start_times))] # list of tuples of lists
    
    tag_pointer = tag_start
    cur_tag_value = tag_times[tag_pointer]
    bucket_pointer = 0
    while tag_pointer < tag_end and bucket_pointer < len(buckets) - 1:
        if not bucket_start_times[bucket_pointer] <= cur_tag_value < bucket_start_times[bucket_pointer + 1]:
            bucket_pointer += 1
            
        buckets[bucket_pointer][0].append(tag_times[tag_pointer] * 1000)
        tag_pointer += 1
        cur_tag_value = tag_times[tag_pointer]

    img_pointer = img_start
    bucket_pointer = 0
    while img_pointer < img_end and bucket_pointer < len(buckets) - 1:
        if not bucket_start_times[bucket_pointer] <= img_times[img_pointer] < bucket_start_times[bucket_pointer + 1]:
            bucket_pointer += 1
            
        buckets[bucket_pointer][1].append(img_times[img_pointer])
        img_pointer += 1
    
    return buckets

def bucket_stats(buckets):
    bucket_sizes = defaultdict(int)
    
    for tag_bucket, img_bucket in buckets:
        bucket_sizes[f'tag_size_{len(tag_bucket)}'] += 1
        bucket_sizes[f'img_size_{len(img_bucket)}'] += 1
        bucket_sizes[f'all_size_{len(tag_bucket) + len(img_bucket)}'] += 1
    
    return bucket_sizes


# In[4]:


experiment_name = 'stationary_center'
rfid_data = rfid_info(f'data/{experiment_name}/tag_data.csv')
img_data = img_metadata(f'data/{experiment_name}/{experiment_name}_meta.txt')
disc1_tag = select_useful_rfid(rfid_data)
disc_time_data = [tag_data['time_millis'] for tag_data in disc1_tag]


# In[15]:


# We create buckets of the same size across ALL readings and images
# So care is needed to determine the bucket size. We want to have a bucket to have 

bucket_size = 0.04

for root, dirs, files in os.walk('data/'):
    for experiment_name in dirs:
        experiment_dir = os.path.join(root, experiment_name)
        try:
            rfid_data = rfid_info(os.path.join(experiment_dir, 'tag_data.csv'))
        except:
            print(f'{experiment_dir} does not have tag_data.csv')
            print()
            continue
        
        try:
            img_data = img_metadata(os.path.join(experiment_dir, f'{experiment_name}_meta.txt'))
        except:
            print(f'{experiment_dir} does not have _meta.txt')
            print()
            continue
            
        disc_tag = select_useful_rfid(rfid_data)
        disc_time_data = [tag_data['time_millis'] for tag_data in disc_tag]
        buckets = bucketize(bucket_size, disc_time_data, img_data)
        for tag_times, img_times in buckets:
            print(tag_times, img_times)
        all_stats = bucket_stats(buckets)

#         for stat_name, count in sorted(all_stats.items()):
#             if not stat_name.startswith('all'):
#                 print(f'{stat_name}: {count}')
#         print()


# E20038EBB5953DC9B7693E9C
# E20038EBB5953F49B7693EA2
