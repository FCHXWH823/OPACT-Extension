import os
import csv
import shutil
import numpy as np
import torch
import pandas as pd
from parameters import *
from generator import * 
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

class CompressorNetwork:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        
        df = pd.read_csv(csv_path)

        # parse basic structure of the compressor network
        self.N_STAGE = df["stage"].max() + 1
        self.N_COL   = df["col"].max() + 1
        self.n_bits  = (self.N_COL + 1) // 2
        self.weight = 2 ** torch.arange(self.N_COL, device=device)

        # init compressors storage
        self.comps = [ [[] for _ in range(self.N_COL)] for _ in range(self.N_STAGE) ]

        # init pin and bit indices between compressors and the final stage with compressors for each column
        self.pin_index = [ [0 for _ in range(self.N_COL)] for _ in range(self.N_STAGE) ]
        self.bit_index = [ [min(j + 1, int(2 * self.n_bits - 1 - j), self.n_bits) for j in range(self.N_COL)] ]+ [[0 for _ in range(self.N_COL)] for _ in range(self.N_STAGE)]
        self.stage_index = [0 for _ in range(self.N_COL)] 

        self.comp_info = {
            "ex-3:2": dict(id=0, pins=3, bits=(2, 1), fn=self.ex32_fn),
            "ex-2:2": dict(id=1, pins=2, bits=(2, 1), fn=self.ex22_fn),
            "ap-3:2": dict(id=2, pins=3, bits=(2, 0), fn=self.ap32_fn),
            "ap-4:2": dict(id=3, pins=4, bits=(2, 0), fn=self.ap42_fn),
            "dummy" : dict(id=4, pins=1, bits=(1, 0), fn=self.dummy_fn),
        }

        def build_from_csv():
            # INFO: stage, col is ordered in the CSV
            expected_pins = [min(j + 1, 2 * self.n_bits - 1 - j, self.n_bits) for j in range(self.N_COL)]

            current_stage, current_col = 0, 0
            used_pins = 0

            def alloc_pins(s, c, n):
                idx = list(range(self.pin_index[s][c], self.pin_index[s][c] + n))
                self.pin_index[s][c] += n
                return idx

            def alloc_bit(s, c, n, carry=False):
                col_out = c + 1 if carry else c
                idx = list(range(self.bit_index[s][col_out], self.bit_index[s][col_out] + n))
                self.bit_index[s][col_out] += n
                return [(col_out, b) for b in idx]
            
            def alloc_bits(s, c, n, carry=False):
                s = s + 1
                if carry:
                    return alloc_bit(s, c, n-1, carry=False)+alloc_bit(s, c, n=1, carry=True)
                return alloc_bit(s, c, n, carry=False)

            for _, row in df.iterrows():
                s, c = int(row["stage"]), int(row["col"])
                comp_type = row["comp_type"]
                count = int(row["count"])
                meta = self.comp_info[comp_type]

                if (s != current_stage) or (c != current_col):
                    # fill dummy for the previous column
                    remaining = expected_pins[current_col] - used_pins
                    if remaining > 0:
                        dummy_meta = self.comp_info["dummy"]
                        for _ in range(remaining):
                            self.comps[current_stage][current_col].append({
                                "type": "dummy",
                                "pins": alloc_pins(current_stage, current_col, dummy_meta["pins"]),
                                "bits": alloc_bits(current_stage, current_col, dummy_meta["bits"][0], carry=False)
                            })
                    if current_stage != s:
                        expected_pins =  self.bit_index[s]
                    current_col, current_stage = c, s
                    used_pins = 0

                for _ in range(count):
                    self.stage_index[c] = s + 1
                    self.comps[s][c].append({
                        "type": comp_type,
                        "pins": alloc_pins(s, c, meta["pins"]),
                        "bits": alloc_bits(s, c, meta["bits"][0], carry=meta["bits"][1])
                    })
                used_pins += meta["pins"] * count

            # deal with the remaining dummy in the last column
            remaining = expected_pins[current_col] - used_pins
            if remaining > 0:
                dummy_meta = self.comp_info["dummy"]
                for _ in range(remaining):
                    self.comps[current_stage][current_col].append({
                        "type": "dummy",
                        "pins": alloc_pins(current_stage, current_col, dummy_meta["pins"]),
                        "bits": alloc_bits(current_stage, current_col, dummy_meta["bits"][0], carry=False)
                    })
        
        # get the info of the compressors from csv
        build_from_csv()
        self._check_valid()

    def _check_valid(self):
        # check pin/bit indices
        for s in range(self.N_STAGE):
            for j in range(self.N_COL):
                if (self.bit_index[s+1][j] < 0):
                    raise ValueError(f"Stage {s} Col {j} has negative output bits.")
                if (self.bit_index[s][j] != self.pin_index[s][j]):
                    raise ValueError(f"Stage {s} Col {j} pin/bit count mismatch: {self.pin_index[s][j]} vs {self.bit_index[s][j]}")
        print(f"[INFO] Compressor network structure valid: {self.N_STAGE} stages, {self.N_COL} columns.")

    #  Bernoulli output polynomials
    @staticmethod
    def ex22_fn(inputs):
        p0, p1 = inputs[:, 0], inputs[:, 1]
        S = p0 + p1 - 2 * p0 * p1
        C = p0 * p1
        return (S, C)

    @staticmethod
    def ex32_fn(inputs):
        p0, p1, p2 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        S = (p0 + p1 + p2) - 2 * (p0*p1 + p0*p2 + p1*p2) + 4 * p0*p1*p2
        C = p0*p1 + p0*p2 + p1*p2 - 2 * p0*p1*p2
        return (S, C)

    @staticmethod
    def ap32_fn(inputs):
        p0, p1, p2 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        S = p0 + p1 - p0*p1
        C = p2 + p0*p1 - p0*p1*p2
        # err = p0 * p1 * p2
        return (S, C)

    @staticmethod
    def ap42_fn(inputs):
        p0, p1, p2, p3 = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
        # S = (p1 + p0 + p3*p2 + 2*p3*p0 + p1*p0
            #  - 2*p3*p1*p0 - p3*p2*p1 - p3*p2*p0 + p3*p2*p1*p0)
        S = (p1 + p0 + p3*p2 - p1*p0
             - p3*p2*p1 - p3*p2*p0 + p3*p2*p1*p0)
        C = (p2 + p3 + p1*p0 - p3*p2 - p2*p1*p0
             - p3*p1*p0 + p3*p2*p1*p0)
        # err = 4 * p0*p1*p2 - 2 * (p0*p1*p2*p3)
        return (S, C)

    @staticmethod
    def dummy_fn(inputs):
        p0 = inputs[:, 0]
        C = torch.zeros_like(p0)
        return (p0, C)
    
class CompressorNetworkTrainer(CompressorNetwork):
    def __init__(self, csv_path="./AC_Allocation_test.csv", lambda_orth=20, num_epochs=500, batch_size=512, lr=1e-1, warmup_epochs=50, log_period=20, eval_period=20, print_period=5, tau_start=0.5, tau_end=0.5, save_dir='Training_log_s2', train_loss_csv='loss.csv', eval_log='eval_diff.txt', final_log='final_diff.txt', conn_log_dir='./conn'):
        super().__init__(csv_path=csv_path)
        
        # === Training hyperparams ===
        self.lambda_orth=lambda_orth
        self.num_epochs = num_epochs
        # self.batch_size = batch_size
        self.total_patterns = (1 << self.n_bits) ** 2
        self.batch_size = min(self.total_patterns, batch_size)
        self.test_batch_size = min(self.total_patterns, batch_size << 4)
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.log_period = log_period
        self.eval_period = eval_period
        self.print_period = print_period
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.device = device
        self.save_dir = f"{save_dir}_col{self.N_COL}"
        self.train_loss_csv = os.path.join(self.save_dir, train_loss_csv)
        self.eval_log = os.path.join(self.save_dir, eval_log)
        self.final_log = os.path.join(self.save_dir, final_log)
        self.conn_log_dir = os.path.join(self.save_dir, conn_log_dir)

        # === Init Param ===
        self._init_param()

        # === Init Baseline (random conn) Param
        self._init_random_connect()

        # === Optimizer ===
        cp_params = [p for stage in self.logits for p in stage if p is not None and p.requires_grad]
        self.optimizer = torch.optim.Adam(cp_params, lr=self.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=self.lr * 0.01)
        # self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)

        # === Temperature schedule (for softmax) ===
        self.temp = self.tau_start  # soft connect -> hard connect over epochs

        print(f"[INIT] Trainer ready on {device} | batch_size={self.batch_size}, epochs={num_epochs}, lr={lr}, tau=({tau_start}→{tau_end})")

        # === Init File Path ===
        self._init_logs()

    # --- τ 退火函数 ---
    def _update_tau(self, epoch):
        # Linear decay from tau_start → tau_end
        # progress = min(epoch / self.num_epochs, 1.0)
        # self.temp = self.tau_start + (self.tau_end - self.tau_start) * progress
        self.temp = max(self.tau_end, self.tau_start * (0.99 ** epoch))
    
    def _update_lambda(self, epoch):
        self.lambda_orth = max(10, self.lambda_orth * 0.995)

    def _init_logs(self):
        os.makedirs(self.save_dir, exist_ok=True)
        if os.path.exists(self.conn_log_dir):
            shutil.rmtree(self.conn_log_dir) 
        os.makedirs(self.conn_log_dir, exist_ok=True)

        with open(self.train_loss_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

        with open(self.eval_log, mode='w') as f:
            f.write("# Evaluation log: records error matrix and MED of random 1v1 connections vs trained connections\n")

        with open(self.final_log, mode='w') as f:
            f.write("# Final evaluation results after full training\n")

        info_log = os.path.join(self.save_dir, "train_info.md")
        with open(info_log, 'w') as f:
            f.write("# Training Info \n\n")
        
            # Training Configuration
            f.write("## Training Configuration\n\n")
            f.write(f"Lambda orth: {self.lambda_orth}\n")
            f.write(f"Batch size: {self.batch_size}\n")
            f.write(f"Number of epochs: {self.num_epochs}\n")
            f.write(f"Learning rate: {self.lr}\n")
            f.write(f"Optimizer: {type(self.optimizer).__name__}\n")
            f.write(f"Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}\n")
            f.write(f"Warmup epochs: {self.warmup_epochs}\n")
            f.write(f"Temperature (tau): {self.temp}\n\n")
            
            # Compressor Network Info
            f.write("## Compressor Network Info\n\n")
            f.write(f"Compressor allocation: {self.csv_path}\n")
            f.write(f"Total patterns: {self.total_patterns}\n")
            f.write(f"Number of stages: {self.N_STAGE}\n")
            f.write(f"Number of columns: {self.N_COL}\n")
            f.write(f"Bits per column: {self.n_bits}\n")
            f.write(f"Final compressor position per column: {self.stage_index}\n")
            f.write(f"Pin index per stage: {self.pin_index}\n")
            f.write(f"Bit index per stage: {self.bit_index}\n\n")
            
            # Loss / logging
            f.write("## Logging Info\n\n")
            f.write(f"Train loss CSV: {self.train_loss_csv}\n")
            f.write(f"Eval log: {self.eval_log}\n")
            f.write(f"Final log: {self.final_log}\n")
            f.write(f"Connection log directory: {self.conn_log_dir}\n\n")

    def _init_param(self):
        self.logits = []
        for s in range(self.N_STAGE):  # stage[i] → stage[i+1]
            stage_cp_logits = []
            for j in range(self.N_COL):
                if self.stage_index[j] > s:
                    param = torch.randn(self.bit_index[s][j], self.pin_index[s][j], requires_grad=True, device=device)
                else:
                    param = None
                stage_cp_logits.append(param)
            self.logits.append(stage_cp_logits)
    
    def _init_random_connect(self):
        self.random_perm = []
        for stage in range(self.N_STAGE):
            stage_perm = []
            for j in range(self.N_COL):
                if self.stage_index[j] <= stage:
                    perm = None
                else:
                    perm = torch.randperm(self.bit_index[stage][j], device=device)
                stage_perm.append(perm)
            self.random_perm.append(stage_perm)

    def _connect_stage(self, pb_stage, stage, Soft, Random_conn):
        """
        Input:
            pb_stage: [B, N_COL, N_BIT] 
            stage: current stage index
            Soft: 1 for train, 0 for eval
            Random_conn: 1 for random conn baseline, 0 for trained param
        Output:
            pp_stage: [B, N_COL, N_PIN] 
        """
        pp_stage = []
        for j in range(self.N_COL):
            if self.stage_index[j] <= stage:
                # no compressor since this stage
                # copy bits to pins directly
                pp_stage.append(pb_stage[j][:, :self.pin_index[stage][j]])
                continue
            if Soft:
                conn_prob = F.softmax(self.logits[stage][j]/self.temp, dim=0)
                pp_col = pb_stage[j] @ conn_prob  # [B, n_pin_j]
            else:
                if Random_conn:
                    max_idx = self.random_perm[stage][j]
                else:
                    max_idx = torch.argmax(self.logits[stage][j], dim=0)
                pp_col = pb_stage[j].gather(1, max_idx.unsqueeze(0).expand(pb_stage[j].size(0), -1))
            pp_stage.append(pp_col)
        return pp_stage

    # def _connect_stage_soft(self, pb_stage, stage):
    #     """
    #     Input:
    #         pb_stage: [B, N_COL, N_BIT] 
    #         stage: current stage index
    #     Output:
    #         pp_stage: [B, N_COL, N_PIN] 
    #     """
    #     # softmax over bit dim to get connection probabilities
    #     pp_stage = []
    #     for j in range(self.N_COL):
    #         if self.stage_index[j] <= stage:
    #             # no compressor since this stage
    #             # copy bits to pins directly
    #             pp_stage.append(pb_stage[j][:, :self.pin_index[stage][j]])
    #             continue
    #         conn_prob = F.softmax(self.logits[stage][j]/self.temp, dim=0)
    #         pp_col = pb_stage[j] @ conn_prob  # [B, n_pin_j]
    #         pp_stage.append(pp_col)
    #     return pp_stage

    # def _connect_stage_hard(self, pb_stage, stage, random_conn = False):
    #     """
    #     Input:
    #         pb_stage: [B, N_COL, N_BIT] 
    #         stage: current stage index
    #     Output:
    #         pp_stage: [B, N_COL, N_PIN] 
    #     """
    #     pp_stage = []
    #     for j in range(self.N_COL):
    #         if self.stage_index[j] <= stage:
    #             # no compressor since this stage
    #             # copy bits to pins directly
    #             pp_stage.append(pb_stage[j][:, :self.pin_index[stage][j]])
    #             continue
    #         if random_conn:
    #             max_idx = self.random_perm[stage][j]
    #         else:
    #             max_idx = torch.argmax(self.logits[stage][j], dim=0)
    #         chosen_bits = pb_stage[j].gather(1, max_idx.unsqueeze(0).expand(pb_stage[j].size(0), -1))
    #         pp_stage.append(chosen_bits)
    #     return pp_stage

    def _compress_stage(self, pp, stage):
        """
        Input:
            pp: [B, N_COL, N_PIN]
            stage: current stage index
        Output:
            pb: [B, N_COL, N_BIT]
        """
        pb_stage = [torch.zeros(pp[0].shape[0], self.bit_index[stage+1][c], device=device) for c in range(self.N_COL)]

        for c in range(self.N_COL):
            col_comps = self.comps[stage][c]
            for comp in col_comps:
                # gather input probabilities (batchified)
                inputs = pp[c][:, comp["pins"]]
                # run compressor
                S, C = self.comp_info[comp["type"]]["fn"](inputs)
                # accumulate results
                pb_stage[comp['bits'][0][0]][:, comp['bits'][0][1]] += S
                # carry bit only for non-dummy compressors
                if comp["type"] != "dummy" and len(comp["bits"]) > 1:
                    pb_stage[comp['bits'][1][0]][:, comp['bits'][1][1]] += C
        return pb_stage
    
    def _mat_to_val(self, bit_value, app):
        """
        Input:
            bit_value: [N_COL[B, N_BIT]]
        Output:
            mean: [B]
            var: [B]
        """
        col_sum = torch.stack([bv.sum(dim=-1) for bv in bit_value], dim=-1)
        col_var_sum = torch.stack([ (bv * (1 - bv)).sum(dim=-1) for bv in bit_value ], dim=-1)
        # col_sum = bit_value.sum(dim=-1)  # [B, N_COL]
        mean = (col_sum * self.weight).sum(dim=-1)
        if not app:
            return mean, torch.zeros_like(mean)
        # col_var_sum = (bit_value * (1 - bit_value)).sum(dim=-1)  # [B, N_COL]
        var = (col_var_sum * (self.weight ** 2)).sum(dim=-1)
        return (mean, var)
    
    def _compute_loss(self, input_batch, Train, Baseline, Log=False, log_path=None):
        '''
        Input:
            input_batch: [B, N_COL, N_PIN]
            epoch: N
            Train: 1 for train, 0 for eval
            Baseline: 1 for random conn baseline, 0 for trained param
            Log: record connection if True, used when evaluate.
        Output:
            loss: tensor
        '''
        loss = 0.0
        pb = [input_batch[:, c, :self.pin_index[0][c]] for c in range(self.N_COL)]  # [N_COL[B, N_PIN]

        acc_value, _ = self._mat_to_val(pb, app=False)

        # propagate
        for i in range(self.N_STAGE):
            pp = self._connect_stage(pb, i, Soft=Train, Random_conn=Baseline)
            pb = self._compress_stage(pp, i)
        
        pred_mean, pred_var = self._mat_to_val(pb, app=Train)
        loss = torch.mean((pred_mean - acc_value) ** 2 + pred_var)  # MSE loss

        # loss
        orth_loss = 0.0
        if Train:
            for stage in self.logits:
                for logit_col in stage:
                    if logit_col is not None:
                        P = F.softmax(logit_col / self.temp, dim=0)  # [bits, pins]
                        I = torch.eye(P.shape[1], device=P.device)
                        orth_loss += torch.norm(P.T @ P - I, p='fro') ** 2
            loss = loss + self.lambda_orth * orth_loss

        if Log:
            if log_path == None:
                log_path = self.eval_log
            self._log_diff(pred_mean - acc_value, log_file=log_path, Baseline=Baseline)
        return loss
    
    def _log_conn(self, epoch):
        log_path = os.path.join(self.conn_log_dir, f"epoch_{epoch:04d}.txt")

        with torch.no_grad():
            logits_stage = [
                [logit_col.detach().cpu() if logit_col is not None else None for logit_col in logit_stage]
                for logit_stage in self.logits
            ]

        with open(log_path, "w") as f:
            f.write("==== Hard Conn ====\n")
            for i, logit_stage in enumerate(logits_stage):
                f.write(f"\n-- Stage {i} --\n")
                for j, logit_col in enumerate(logit_stage):
                    if logit_col is None:
                        f.write(f"Col {j} : No compressor\n")
                        continue
                    max_idx = torch.argmax(logit_col, dim=0)
                    f.write(f"Col {j} : ")
                    f.write(f"{max_idx.numpy().astype(int)}\n")

    def _log_diff(self, delta, log_file, Baseline=False):
        np.set_printoptions(threshold=np.inf)
        with open(log_file, "a") as f:
            if Baseline:
                f.write(f"\n==== Random 1v1 Connection Evaluation ====\n")
            else:
                f.write(f"\n==== Trained Connection Evaluation ====\n")
            diff = delta.detach().cpu().numpy()
            f.write("pred-acc:\n")
            f.write(f"{diff}\n")
            f.write(f"MED: {torch.mean(torch.abs(delta)).item():.6f}\n")

    def train(self):
        # init DataLoader
        loader = IP_generator(self.n_bits, batch=self.batch_size, exhaustive=False)
        print(f"[START TRAINING]")

        for epoch in range(self.num_epochs):
            # get input pattern batch [B, N_COL, N_PIN]
            input_batch = next(loader)

            # self._update_tau(epoch)
            loss = self._compute_loss(input_batch, Train=True, Baseline=False)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch < self.warmup_epochs:
                warmup_lr = self.lr * (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            else:
                self.scheduler.step()

            # print
            if epoch % self.print_period == 0 or epoch == self.num_epochs - 1:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"[TRAIN {epoch}] Train Loss={loss.item():.6f}, LR={current_lr:.6f}, Tau={self.temp:.4f}")

            # record connection
            if epoch % self.log_period == 0 or epoch == self.num_epochs - 1:
                self._log_conn(epoch)

            # EVAL
            if epoch % self.eval_period == 0 or epoch == self.num_epochs - 1:
                # Record eval diff and random conn diff
                with open(self.eval_log, "a") as f:
                    f.write(f"\n-- Epoch {epoch} --\n")
                with torch.no_grad():
                    # input_batch = next(loader)
                    eval_loss = self._compute_loss(input_batch, Train=False, Baseline=False, Log=True)
                    rand_loss = self._compute_loss(input_batch, Train=False, Baseline=True, Log=True)
                # print(f"##################[Eval {epoch}] Eval Loss={eval_loss.item():.6f}, Random Conn Loss={rand_loss.item():.6f}###################")
                # record train loss and eval loss
                with open(self.train_loss_csv, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, loss.item(), eval_loss.item()])
            else:
                with open(self.train_loss_csv, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, loss.item(), None])

    def test(self):
        loader = IP_generator(self.n_bits, batch=self.test_batch_size, exhaustive=True)
        print(f"[START TESTING]")

        # get input pattern batch [B, N_COL, N_PIN]
        input_batch = next(loader)
    
        with torch.no_grad():
            eval_loss = self._compute_loss(input_batch, Train=False, Baseline=False, Log=True, log_path=self.final_log)
            rand_loss = self._compute_loss(input_batch, Train=False, Baseline=True, Log=True, log_path=self.final_log)

        print(f"[TEST] Test Loss={eval_loss.item():.6f}, Random Conn Loss={rand_loss.item():.6f}")
                    
                
        




    # def _compute_random_conn_loss(self, input_batch):
        
    #     # if compare:
    #     #     print(f"[COMPARE] Start comparing random 1v1 connection with trained connection...")
    #     # print(f"[BASELINE] Start evaluating random connection...")

    #     pb = [input_batch[:, c, :self.pin_index[0][c]] for c in range(self.N_COL)]  # [N_COL[B, N_PIN]
    #     acc_value, _ = self._mat_to_val(pb, False)
    #     # propagate
    #     for i in range(self.N_STAGE):
    #         pp = self._connect_stage_hard(pb, i, True)
    #         pb = self._compress_stage(pp, i)
        
    #     pred, _ = self._mat_to_val(pb, False)

    #     self._rec_diff(pred - acc_value, trained=False)
    #     loss = torch.mean((pred - acc_value)**2)

    #     return loss
    

    # def _compress_stage(self, pp, stage):
    #     """
    #     Input:
    #         pp: [B, N_COL, N_PIN]
    #         stage: current stage index
    #     Output:
    #         pb: [B, N_COL, N_BIT]
    #     """
    #     pb_stage = [torch.zeros(self.batch_size, self.bit_index[stage][c], device=device) for c in range(self.N_COL)]

    #     for c in range(self.N_COL):
    #         col_comps = self.comps[stage][c]
    #         for comp in col_comps:
    #             # gather input probabilities (batchified)
    #             inputs = pp[c][:, comp["pins"]]
    #             # run compressor
    #             S, C = self.comp_info[comp["type"]]["fn"](inputs)
    #             # accumulate results
    #             pb_stage[comp['bits'][0][0]][:, comp['bits'][0][1]] += S
    #             # carry bit only for non-dummy compressors
    #             if comp["type"] != "dummy" and len(comp["bits"]) > 1:
    #                 pb_stage[comp['bits'][1][0]][:, comp['bits'][1][1]] += C
    #     return pb_stage

        # INFO: can be optimized by dealing with dummy first.
        # for j in range(N_COL):
        #     pp_col = pp_stage[j]
        #     for comp in comp_stage[j]:
        #         inputs = [pp_col[i] for i in comp['pins']]
        #         (S, C), comp_err_tmp = F_COMPRESSOR[comp['type']](inputs)
        #         pb_stage[comp['bits'][0][0]][comp['bits'][0][1]] += S
        #         if comp['type'] < 4:  # only add carry for non-dummy compressors
        #             pb_stage[comp['bits'][1][0]][comp['bits'][1][1]] += C

        #         comp_err[j] += comp_err_tmp

        # return pb_stage, comp_err

    


# F_COMPRESSOR = [
#     lambda inputs: ex32_fn(*inputs),
#     lambda inputs: ex22_fn(*inputs),
#     lambda inputs: ap32_fn(*inputs),
#     lambda inputs: ap42_fn(*inputs),
#     lambda inputs: dummy_fn(*inputs),
# ]

# COMPRESSOR_LIBRARY = {
#     "ex-3:2": {"id": 0, "pins": 3, "bits": [(1, False), (1, True)], "fn": ex32_fn},
#     "ex-2:2": {"id": 1, "pins": 2, "bits": [(1, False), (1, True)], "fn": ex22_fn},
#     "ap-3:2": {"id": 2, "pins": 3, "bits": [(2, False)],"fn": ap32_fn},
#     "ap-4:2": {"id": 3, "pins": 4, "bits": [(2, False)],"fn": ap42_fn},
#     "dummy":  {"id": 4, "pins": 1, "bits": [(1, False)],"fn": dummy_fn}
# }
