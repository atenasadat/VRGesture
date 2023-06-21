from tqdm import tqdm

from data import Data


# modalities = ['pose/data', 'audio/log_mel_512', 'text/bert']
# path = 'VRGesture/data/processed/minhaj/cmu0000024058.h5'
# import h5py
# x = h5py.File(path, 'a')
# print(x.keys())
#
# for modality in modalities:
#     try:
#         h5 = h5py.File(path, 'a')
#     except:
#         print("error : {} file with madality {}".format(path, modality))

"""
cmu0000024058

cmu0000024338
cmu0000024446
cmu0000024447
cmu0000024449
cmu0000024450
cmu0000024451
"""
# common_kwargs = dict(path2data ="VRGesture/data",
#                      speaker = ['minhaj'],#one or more speaker names
#                      modalities = ['pose/data', 'audio/log_mel_512', 'text/bert'],
#                      fs_new = [15, 15, 15], #list of frame rates for each modality in modalities
#                      batch_size = 32,
#                      window_hop = 5 #number of frames a window hops in an interval to contruct samples.
#                      )
#
# data = Data(**common_kwargs)
#
#
#
#
# for batch in data.train:
#     print("-------")
#     for key in batch.keys():
#         if key != 'meta':
#             print('{}: {}'.format(key, batch[key].shape))
#     print()


""""
    
    pose/data: torch.Size([32, 64, 104]) pose/data" has 104 dimensions which is the same as 52 joints with XY coordinates
    audio/log_mel_512: torch.Size([32, 64, 128])
    text/bert: torch.Size([32, 64, 768])
    text/token_duration: torch.Size([32, 19]) hape of "text/token_duration" implies the maximum length of a sentence in this mini-batch is 19
    text/token_count: torch.Size([32])
    style: torch.Size([32, 64]) is the relative style id of the speakers in the dataset
    idx: torch.Size([32]) refers to the idx of the object of the Data clas
"""
#
# common_kwargs.update(dict(modalities = ['pose/data', 'audio/log_mel_512', 'text/tokens'],
#                          repeat_text = 0))
#
# data_n = Data(**common_kwargs)


data = Data(path2data='data/', speaker='minhaj')
data_train = data.train
data_dev = data.dev
data_test = data.test

print('Data Loaded')

print(type(data_train))