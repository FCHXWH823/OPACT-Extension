# Training Info 

## Training Configuration

Batch size: 512
Number of epochs: 800
Learning rate: 0.01
Optimizer: Adam
Scheduler: CosineAnnealingLR
Warmup epochs: 20
Temperature (tau): 2.0

## Compressor Network Info

Compressor allocation: ./stage2/AC_Allocation_6240.json
Total patterns: 4294967296
Number of stages: 5
Number of columns: 35
Bits per column: 16
Final compressor position per column: [0, 0, 1, 1, 2, 4, 4, 3, 3, 3, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0]
Pin index per stage: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0], [1, 2, 2, 1, 5, 4, 3, 7, 3, 6, 6, 10, 4, 8, 8, 7, 9, 7, 8, 6, 7, 7, 10, 8, 5, 4, 4, 4, 2, 2, 2, 0, 0, 0, 0], [1, 2, 2, 1, 2, 4, 2, 3, 3, 3, 4, 5, 5, 2, 4, 5, 4, 5, 4, 5, 4, 5, 7, 6, 7, 3, 3, 3, 3, 2, 2, 1, 0, 0, 0], [1, 2, 2, 1, 2, 3, 2, 2, 2, 2, 1, 5, 2, 3, 1, 3, 2, 3, 2, 3, 2, 3, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 0, 0, 0], [1, 2, 2, 1, 2, 2, 1, 3, 2, 2, 1, 3, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0]]
Bit index per stage: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0], [1, 2, 2, 1, 5, 4, 3, 7, 3, 6, 6, 10, 4, 8, 8, 7, 9, 7, 8, 6, 7, 7, 10, 8, 5, 4, 4, 4, 2, 2, 2, 0, 0, 0, 0], [1, 2, 2, 1, 2, 4, 2, 3, 3, 3, 4, 5, 5, 2, 4, 5, 4, 5, 4, 5, 4, 5, 7, 6, 7, 3, 3, 3, 3, 2, 2, 1, 0, 0, 0], [1, 2, 2, 1, 2, 3, 2, 2, 2, 2, 1, 5, 2, 3, 1, 3, 2, 3, 2, 3, 2, 3, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 0, 0, 0], [1, 2, 2, 1, 2, 2, 1, 3, 2, 2, 1, 3, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0], [1, 2, 2, 1, 2, 2, 1, 3, 2, 2, 1, 3, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0]]

## Logging Info

Train loss CSV: Training_log_s2_ac_large_sinkhorn_3_col35/loss.csv
Eval log: Training_log_s2_ac_large_sinkhorn_3_col35/eval_diff.txt
Final log: Training_log_s2_ac_large_sinkhorn_3_col35/final_diff.txt
Connection log directory: Training_log_s2_ac_large_sinkhorn_3_col35/./conn

