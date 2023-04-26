import os
import torch
import torch.nn as nn
import time
from torch.distributed.rpc import init_rpc
from drecg.feature_extraction.distributed import define_model_for_tune


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
init_rpc('worker', rank=0, world_size=1)


devices_list = [torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2"), torch.device("cuda:3")]
model_pipe = define_model_for_tune(devices_list, microbatch_num=1)

loss_fn = nn.BCEWithLogitsLoss()
y_true = torch.ones(32, 1, dtype=torch.float32).to(devices_list[-1])

model_pipe.train()
adam_w = torch.optim.AdamW(model_pipe.parameters(), lr=1e-3)

for i in range(40):
    init_time = time.time()
    dummy_img_a = torch.rand(32, 3, 224, 224).to(devices_list[0])
    dummy_img_b = torch.rand(32, 3, 224, 224).to(devices_list[0])
    adam_w.zero_grad()
    out = model_pipe((dummy_img_a, dummy_img_b))
    out = out.local_value()
    loss = loss_fn(out, y_true)
    loss.backward()
    adam_w.step()
    total_time_segs = time.time() - init_time
    print("iter: {}, loss: {}, time: {} segs".format(i, loss.item(), total_time_segs))
