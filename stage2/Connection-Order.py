from generator import *
from Compressor_Network import *

net = CompressorNetworkTrainer(csv_path="./AC_Allocation_update.csv", lambda_orth=2000, num_epochs=500, batch_size=128, lr=1e-1, warmup_epochs=50, log_period=20, eval_period=20, print_period=5, tau_start=0.5, tau_end=0.5, save_dir='Training_log_s2_', train_loss_csv='loss.csv', eval_log='eval_diff.txt', final_log='final_diff.txt', conn_log_dir='./conn')
# net.init_param()
net.train()
net.test()
# for j, col in enumerate(comp[0]):
#     print(f"Column {j}:")
#     for c in col:
#         print(c)

# 启动训练
# train(comp, logits, num_steps=200, batch_size=64, lr=1e-2)
