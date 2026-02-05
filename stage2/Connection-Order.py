from generator import *
from Compressor_Network import *
num_epochs=800
schedule_cfg = {
    "tau": {
        "start": 2.0,
        "v0": 1.5,
        "v1": 0.5,
        "end": 0.1,
        "t0": 100,
        "t1": 400,
        "type0": "linear",
        "type1": "cosine",
        "type2": "exp"
    },
    "lambda_orth": {
        "start": 0.0,   
        "v0": 0.0,
        "v1": 1e3,
        "end": 5e5,
        "t0": 100,
        "t1": 400,
        "type0": "constant",
        "type1": "exp",
        "type2": "cosine",
        "entropy_threshold": 130.0
    }
}

net = CompressorNetworkTrainer(file_path="./stage2/AC_Allocation_6240.json", schedule_cfg=schedule_cfg, num_epochs=num_epochs, batch_size=512, test_batch_size=1<<16, lr=1e-2, warmup_epochs=20, log_period=20, eval_period=5, print_period=1, save_dir='Training_log_s2_ac_large_sinkhorn_3', train_loss_csv='loss.csv', eval_log='eval_diff.txt', final_log='final_diff.txt', conn_log_dir='./conn')
net.train()
net.test()
# tau_end=0.01
# lambda_orth=1000000