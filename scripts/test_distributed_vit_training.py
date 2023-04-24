import os
import torch
import torch.nn as nn
import time
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)
from drecg.feature_extraction.distributed import sequential_model_to_devices

from torch.distributed.pipeline.sync import Pipe
from drecg.feature_extraction.distributed import VitImageFeatureExtractor
model = VitImageFeatureExtractor.load_pretrained()

model_seq = model.to_sequential()
model_seq.add_module('head', nn.Linear(1280, 1))

device0 = torch.device("cuda:0")
device1 = torch.device("cpu")
sequential_model_to_devices(model_seq, device0, device1)
model_pipe = Pipe(model_seq, chunks=2)


loss_fn = nn.BCEWithLogitsLoss()
y_true = torch.ones(10, 1, dtype=torch.float32)

model_pipe.train()
adam_w = torch.optim.AdamW(model_pipe.parameters(), lr=1e-3)

for i in range(2):
    init_time = time.time()
    dummy_tensor_input = torch.rand(10, 3, 224, 224)
    dummy_tensor_input = dummy_tensor_input.to(device0)
    adam_w.zero_grad()
    out = model_pipe(dummy_tensor_input)
    out = out.local_value()
    loss = loss_fn(out, y_true)
    loss.backward()
    adam_w.step()
    total_time_segs = time.time() - init_time
    print("iter: {}, loss: {}, time: {} segs".format(i, loss.item(), total_time_segs))