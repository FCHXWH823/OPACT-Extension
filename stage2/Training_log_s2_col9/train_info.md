# Training Info 

## Training Configuration

Lambda orth: 2000
Batch size: 128
Number of epochs: 500
Learning rate: 0.1
Optimizer: Adam
Scheduler: CosineAnnealingLR
Warmup epochs: 50
Temperature (tau): 0.5

## Compressor Network Info

Compressor allocation: ./AC_Allocation_test.csv
Total patterns: 1024
Number of stages: 4
Number of columns: 9
Bits per column: 5
Final compressor position per column: [0, 1, 1, 2, 3, 3, 1, 1, 0]
Pin index per stage: [[1, 2, 3, 4, 5, 4, 3, 2, 1], [1, 1, 2, 3, 4, 3, 2, 2, 2], [1, 1, 2, 1, 3, 3, 2, 2, 2], [1, 1, 2, 1, 2, 2, 2, 2, 2]]
Bit index per stage: [[1, 2, 3, 4, 5, 4, 3, 2, 1], [1, 1, 2, 3, 4, 3, 2, 2, 2], [1, 1, 2, 1, 3, 3, 2, 2, 2], [1, 1, 2, 1, 2, 2, 2, 2, 2], [1, 1, 2, 1, 2, 2, 2, 2, 2]]

## Logging Info

Train loss CSV: Training_log_s2_col9/loss.csv
Eval log: Training_log_s2_col9/eval_diff.txt
Final log: Training_log_s2_col9/final_diff.txt
Connection log directory: Training_log_s2_col9/./conn

