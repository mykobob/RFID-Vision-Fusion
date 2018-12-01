
# coding: utf-8

# In[21]:


import csv
from collections import defaultdict
import math
import pprint as pp
import numpy as np
import os
import matplotlib.image as mpimg


# In[22]:


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


# In[35]:


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
    
class Bucket:
    def __init__(self):
        self.tags = []
        self.imgs = []
        
    def add_tag_data(self, tag_dict):
        impt_info = {"phase_diff": tag_dict["phase_diff"], "time_millis": tag_dict["time_millis"] * 1000}
        self.tags.append(impt_info)
        
    def get_tag_data(self):
        return self.tags
    
    def num_tags(self):
        return len(self.tags)
        
    def add_img_data(self, img_path, time_stamp):
        img = mpimg.imread(img_path)
        all_img_data = {"time_millis": time_stamp, "img": img}
        self.imgs.append(all_img_data)
    
    def get_img_data(self):
        return self.imgs
    
    def num_imgs(self):
        return len(self.imgs)

def bucketize(bucket_size, tag_info, img_times, experiment_root):
    tag_times = [tag_data['time_millis'] for tag_data in tag_info]
    tag_start, tag_end, img_start, img_end = bound_data(tag_times, img_times)
    
    all_buckets_start = min(tag_times[tag_start], img_times[img_start]) - bucket_size / 2
    all_buckets_end = max(tag_times[tag_end], img_times[img_end]) + bucket_size / 2
    num_buckets = math.ceil((all_buckets_end - all_buckets_start) / bucket_size)
    bucket_start_times = [all_buckets_start + bucket_size * i for i in range(0, num_buckets)]
    buckets = [Bucket() for i in range(len(bucket_start_times))] # list of tuples of lists
    
    tag_pointer = tag_start
    cur_tag_value = tag_times[tag_pointer]
    bucket_pointer = 0
    while tag_pointer < tag_end and bucket_pointer < len(buckets) - 1:
        if not bucket_start_times[bucket_pointer] <= cur_tag_value < bucket_start_times[bucket_pointer + 1]:
            bucket_pointer += 1
            
        buckets[bucket_pointer].add_tag_data(tag_info[tag_pointer])
        tag_pointer += 1
        cur_tag_value = tag_times[tag_pointer]

    img_pointer = img_start
    bucket_pointer = 0
    while img_pointer < img_end and bucket_pointer < len(buckets) - 1:
        if not bucket_start_times[bucket_pointer] <= img_times[img_pointer] < bucket_start_times[bucket_pointer + 1]:
            bucket_pointer += 1
            
#         buckets[bucket_pointer][1].append(img_times[img_pointer])
        buckets[bucket_pointer].add_img_data(os.path.join(experiment_root, 'run_{}.png'.format(img_pointer)), img_times[img_pointer])
        img_pointer += 1
    
    return buckets

def bucket_stats(buckets):
    bucket_sizes = defaultdict(int)
    
    for bucket in buckets:
        bucket_sizes[f'tag_size_{bucket.num_tags()}'] += 1
        bucket_sizes[f'img_size_{bucket.num_imgs()}'] += 1
        bucket_sizes[f'all_size_{bucket.num_tags() + bucket.num_imgs()}'] += 1
    
    return bucket_sizes


# In[4]:


experiment_name = 'stationary_center'
rfid_data = rfid_info(f'data/{experiment_name}/tag_data.csv')
img_data = img_metadata(f'data/{experiment_name}/{experiment_name}_meta.txt')
disc1_tag = select_useful_rfid(rfid_data)
disc_time_data = [tag_data['time_millis'] for tag_data in disc1_tag]


# In[42]:


# We create buckets of the same size across ALL readings and images
# So care is needed to determine the bucket size. We want to have a bucket to have 

bucket_size = 0.04

for root, dirs, files in os.walk('data/'):
    for experiment_name in dirs:
        if experiment_name != 'center_away':
            continue
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
            
        disc_tag_info = select_useful_rfid(rfid_data)
#         disc_time_data = [tag_data['time_millis'] for tag_data in disc_tag]
        buckets = bucketize(bucket_size, disc_tag_info, img_data, experiment_dir)
#         for bucket in buckets:
#             print(bucket.get_tag_data(), bucket.get_img_data())
        all_stats = bucket_stats(buckets)

#         for stat_name, count in sorted(all_stats.items()):
#             if not stat_name.startswith('all'):
#                 print(f'{stat_name}: {count}')
#         print()
        
    break


# E20038EBB5953DC9B7693E9C
# E20038EBB5953F49B7693EA2
