from psutil import CONN_CLOSE
from torch.utils.tensorboard import SummaryWriter
import plotly.graph_objects as go
from PIL import Image
import io

import multi_input_pairs as pairs
from pathlib import Path
import json

import os
import csv
import shutil
import numpy as np
import torch
import pandas as pd
from parameters import *
from generator import * 
from sinkhorn import *
from scheduler import *
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

class CompressorNetwork:
    def __init__(self, file_path):
        self.file_path = file_path

        # self.comp_info = {
        #     "ex-3:2": dict(id=0, pins=3, bits=(2, 1), fn=self.ex32_fn),
        #     "ex-2:2": dict(id=1, pins=2, bits=(2, 1), fn=self.ex22_fn),
        #     "ap-3:2": dict(id=2, pins=3, bits=(2, 0), fn=self.ap32_fn),
        #     "ap-4:2": dict(id=3, pins=4, bits=(2, 0), fn=self.ap42_fn),
        #     "dummy" : dict(id=4, pins=1, bits=(1, 0), fn=self.dummy_fn),
        # }
        
        # get the info of the compressors from csv
        self._get_comp_info()
        self._build_from_json()
        self._check_valid()

    def _get_comp_info(self):
        lib_path = Path("./ACs/ac_lib.json")

        with lib_path.open("r", encoding="utf-8") as f:
            lib_meta = json.load(f)  # dict {comp_name: {...}}

        AC_NAMES = sorted(lib_meta)

        self.comp_info = {}

        for i, name in enumerate(AC_NAMES):
            meta = lib_meta[name]
            info = {}
            
            info["input_ports"] = meta.get("input_ports", [])
            info["npins"] = len(meta.get("input_ports", []))
            
            info["output_ports"] = meta.get("output_ports", [])
            info["nbits"] = len(info["output_ports"])
            info["carry"] = any(p.upper() == "C" for p in info["output_ports"])


            info["F_S"] = pairs.F_S_LIST[i]
            info["F_C"] = pairs.F_C_LIST[i]
            self.comp_info[name] = info
        self.comp_info["dummy"] = {"npins": 1, "nbits": 1, "carry": False, "F_S": pairs.F_S_LIST[-1], "F_C": pairs.F_C_LIST[-1]}

    def _alloc_pins(self, s, c, n):
        idx = list(range(self.pin_index[s][c], self.pin_index[s][c] + n))
        self.pin_index[s][c] += n
        return idx

    def _alloc_bits(self, s, c, n, carry=False):
        s = s+1
        if carry:
            self.bit_index[s][c]+=1
            self.bit_index[s][c+1]+=1
            return [(c, self.bit_index[s][c]-1), (c+1, self.bit_index[s][c+1]-1)]
        else:
            idx = list(range(self.bit_index[s][c], self.bit_index[s][c] + n))
            self.bit_index[s][c] += n
            return [(c, b) for b in idx]
    
    def _build_from_csv(self):
        # !BUG (N_COL > 2*n_bits - 1, carry bits)
        df = pd.read_csv(self.file_path)
        expected_pins = [min(j + 1, 2 * self.n_bits - 1 - j, self.n_bits) for j in range(self.N_COL)]

        current_stage, current_col = 0, 0
        used_pins = 0

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
                            "pins": self._alloc_pins(current_stage, current_col, dummy_meta["pins"]),
                            "bits": self._alloc_bits(current_stage, current_col, dummy_meta["bits"][0], carry=False)
                        })
                if current_stage != s:
                    expected_pins =  self.bit_index[s]
                current_col, current_stage = c, s
                used_pins = 0

            for _ in range(count):
                self.stage_index[c] = s + 1
                self.comps[s][c].append({
                    "type": comp_type,
                    "pins": self._alloc_pins(s, c, meta["pins"]),
                    "bits": self._alloc_bits(s, c, meta["bits"][0], carry=meta["bits"][1])
                })
            used_pins += meta["pins"] * count

        # deal with the remaining dummy in the last column
        remaining = expected_pins[current_col] - used_pins
        if remaining > 0:
            dummy_meta = self.comp_info["dummy"]
            for _ in range(remaining):
                self.comps[current_stage][current_col].append({
                    "type": "dummy",
                    "pins": self._alloc_pins(current_stage, current_col, dummy_meta["pins"]),
                    "bits": self._alloc_bits(current_stage, current_col, dummy_meta["bits"], carry=False)
                })
    
    def _build_from_json(self):
        json_path = Path(self.file_path)
        with json_path.open("r", encoding="utf-8") as f:
            json_data = json.load(f)   
        stage_keys = sorted(json_data.keys(), key=lambda x: int(x.replace("stage", "")))
        self.N_STAGE = len(stage_keys)

        self.N_COL = len(json_data[stage_keys[0]])
        for col_info in json_data[stage_keys[0]]:
            if col_info["bits"] == 0:
                self.N_WIDTH = col_info["col_idx"]
                break
        self.n_bits  = (self.N_WIDTH + 1) // 2
        self.weight = 2 ** torch.arange(self.N_COL, device=device)

        # init compressors storage
        self.comps = [ [[] for _ in range(self.N_COL)] for _ in range(self.N_STAGE) ]

        # init pin and bit indices between compressors and the final stage with compressors for each column
        self.pin_index = [ [0 for _ in range(self.N_COL)] for _ in range(self.N_STAGE) ]
        self.bit_index = [ [max(min(j + 1, int(2 * self.n_bits - 1 - j), self.n_bits), 0) for j in range(self.N_COL)] ]+ [[0 for _ in range(self.N_COL)] for _ in range(self.N_STAGE)]
        self.stage_index = [0 for _ in range(self.N_COL)] 
        
        for stage_key in stage_keys:
            s = int(stage_key.replace("stage", ""))
            for col_info in json_data[stage_key]:
                c = col_info["col_idx"]
                bits = col_info["bits"]
                alloc_dict = col_info["alloc"]
                for comp_type, count in alloc_dict.items():
                    if count == 0:
                        continue
                    if comp_type != "dummy":
                        self.stage_index[c] = s + 1
                    meta = self.comp_info[comp_type]
                    for _ in range(count):
                        self.comps[s][c].append({
                            "type": comp_type,
                            "pins": self._alloc_pins(s, c, meta["npins"]),
                            "bits": self._alloc_bits(s, c, meta["nbits"], carry=meta["carry"])
                        })
        

    def _check_valid(self):
        # check pin/bit indices
        for s in range(self.N_STAGE):
            for j in range(self.N_COL):
                if (self.bit_index[s+1][j] < 0):
                    raise ValueError(f"Stage {s} Col {j} has negative output bits.")
                if (self.bit_index[s][j] != self.pin_index[s][j]):
                    raise ValueError(f"Stage {s} Col {j} pin/bit count mismatch: {self.pin_index[s][j]} vs {self.bit_index[s][j]}")
        print(f"[INFO] Compressor network structure valid: {self.N_STAGE} stages, {self.n_bits} bits, {self.N_COL} columns.")

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
    def __init__(self, schedule_cfg, file_path="./AC_Allocation_test.csv", num_epochs=500, batch_size=512, test_batch_size=2048, lr=1e-1, warmup_epochs=50, log_period=20, eval_period=20, print_period=5, save_dir='Training_log_s2', train_loss_csv='loss.csv', eval_log='eval_diff.txt', final_log='final_diff.txt', conn_log_dir='./conn'):
        super().__init__(file_path=file_path)
        print("Using device:", device)
        # === Training hyperparams ===
        self.num_epochs = num_epochs
        # self.batch_size = batch_size
        self.total_patterns = (1 << self.n_bits) ** 2
        self.batch_size = min(self.total_patterns, batch_size)
        # self.test_batch_size = min(self.total_patterns, max(2048, np.sqrt(self.total_patterns)))
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.log_period = log_period
        self.eval_period = eval_period
        self.print_period = print_period
        self.device = device
        self.save_dir = f"{save_dir}_col{self.N_COL}"
        self.train_loss_csv = os.path.join(self.save_dir, train_loss_csv)
        self.eval_log = os.path.join(self.save_dir, eval_log)
        self.final_log = os.path.join(self.save_dir, final_log)
        self.conn_log_dir = os.path.join(self.save_dir, conn_log_dir)
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, "runs"))
        # === Init Param ===
        self._init_param()

        # === Init Baseline (random conn) Param
        self._init_random_connect()

        # === Optimizer ===
        cp_params = [p for stage in self.logits for p in stage if p is not None and p.requires_grad]
        self.optimizer = torch.optim.Adam(cp_params, lr=self.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=self.lr * 0.01)
        # self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)

        # === Schedulers ===
        self.tau_scheduler = ThreePhaseScheduler(schedule_cfg["tau"], num_epochs)
        self.lambda_scheduler = ThreePhaseScheduler(schedule_cfg["lambda_orth"], num_epochs)

        self.temp = self.tau_scheduler.get(0)
        self.lambda_orth = self.lambda_scheduler.get(0)

        print(f"[INIT] Trainer ready on {device} | batch_size={self.batch_size}, epochs={num_epochs}, lr={lr})")

        # === Init File Path ===
        self._init_logs()

    # --- τ 退火函数 ---
    # def _update_tau(self, epoch):
    #     # Linear decay from tau_start → tau_end
    #     # progress = min(epoch / self.num_epochs, 1.0)
    #     # self.temp = self.tau_start + (self.tau_end - self.tau_start) * progress
    #     if epoch > 150:
    #         self.temp = max(self.tau_end, self.tau_start * (0.99 ** (epoch-150)))
    
    # def _update_lambda(self, epoch):
    #     if epoch > 100:
    #         self.lambda_orth = min(self.lambda_orth_start * (1.1**(epoch-100)), 100000)

    def _update_schedules(self, epoch, entropy=None):
        self.temp = self.tau_scheduler.get(epoch)

        if entropy is None:
            self.lambda_orth = self.lambda_scheduler.get(epoch)
        else:
            self.lambda_orth = self.lambda_scheduler.get(epoch, entropy=entropy)

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
            # f.write(f"Lambda orth: {self.lambda_orth_start}\n")
            f.write(f"Batch size: {self.batch_size}\n")
            f.write(f"Number of epochs: {self.num_epochs}\n")
            f.write(f"Learning rate: {self.lr}\n")
            f.write(f"Optimizer: {type(self.optimizer).__name__}\n")
            f.write(f"Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}\n")
            f.write(f"Warmup epochs: {self.warmup_epochs}\n")
            f.write(f"Temperature (tau): {self.temp}\n\n")
            
            # Compressor Network Info
            f.write("## Compressor Network Info\n\n")
            f.write(f"Compressor allocation: {self.file_path}\n")
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
                if self.stage_index[j] > s and self.bit_index[s][j] != 0:
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
                if self.stage_index[j] <= stage or self.bit_index[stage][j] == 0:
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
            if self.stage_index[j] <= stage or self.bit_index[stage][j] == 0:
                # no compressor since this stage
                # copy bits to pins directly
                pp_stage.append(pb_stage[j][:, :self.pin_index[stage][j]])
                continue
            if Soft:
                # max_idx = torch.argmax(self.logits[stage][j], dim=0)
                # pp_col_hard = pb_stage[j].gather(1, max_idx.unsqueeze(0).expand(pb_stage[j].size(0), -1))
                # conn_prob = F.softmax(self.logits[stage][j]/self.temp, dim=0)
                conn_prob = gumbel_sinkhorn(
                    self.logits[stage][j],
                    tau=self.temp,
                    n_iters=10
                )[0]
                pp_col = pb_stage[j] @ conn_prob  # [B, n_pin_j]
                # pp_col = pp_col_hard - pp_col_soft.detach() + pp_col_soft
            else:
                if Random_conn:
                    max_idx = self.random_perm[stage][j]
                else:
                    P = sinkhorn_log(self.logits[stage][j] / self.temp, 10)[0]
                    max_idx = torch.argmax(P, dim=0)
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
                S = self.comp_info[comp["type"]]["F_S"](inputs)
                C = self.comp_info[comp["type"]]["F_C"](inputs)
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
        pb = [
            (
                input_batch[:, c, :self.pin_index[0][c]]
                if c < self.N_WIDTH
                else torch.zeros(input_batch.shape[0], self.pin_index[0][c],
                                device=input_batch.device,
                                dtype=input_batch.dtype)
            )
            for c in range(self.N_COL)
        ]

        acc_value, _ = self._mat_to_val(pb, app=False)

        # propagate
        for i in range(self.N_STAGE):
            pp = self._connect_stage(pb, i, Soft=Train, Random_conn=Baseline)
            pb = self._compress_stage(pp, i)
        
        pred_mean, pred_var = self._mat_to_val(pb, app=Train)
        conn_loss = torch.mean(torch.abs(pred_mean - acc_value))  # MSE loss
        loss = loss + conn_loss

        # loss
        orth_loss = 0.0
        etp_loss = 0.0
        row_sum_loss = 0.0
        if Train:
            for stage in self.logits:
                for logit_col in stage:
                    if logit_col is not None:
                        # P = F.softmax(logit_col / self.temp, dim=0)  # [bits, pins]
                    
                        # forward 用 P_hard
                        # backward 用 softmax/temperature 来估算梯度
                        # P_soft = F.softmax(logit_col / self.temp, dim=0)
                        P_soft = gumbel_sinkhorn(
                            logit_col,
                            tau=self.temp,
                            n_iters=10
                        )[0]
                        argmax_idx = torch.argmax(P_soft, dim=0)
                        P_hard = torch.zeros_like(logit_col)
                        P_hard[argmax_idx, range(logit_col.shape[1])] = 1
                        P = P_hard - P_soft.detach() + P_soft
                        etp_loss -= (P_soft * torch.log(P_soft + 1e-9)).sum(dim=0).mean()
                        I = torch.eye(P_soft.shape[1], device=P_soft.device)
                        orth_loss += torch.norm(P_soft.T @ P_soft - I, p='fro') ** 2
                        row_sum = P_hard.sum(dim=1)
                        row_sum_loss += (torch.abs(row_sum - 1.0)).sum()
            loss = loss + self.lambda_orth * orth_loss # + self.lambda_orth * etp_loss
        if Log:
            if log_path == None:
                log_path = self.eval_log
            # self._log_diff(pred_mean - acc_value, log_file=log_path, Baseline=Baseline)
        return (loss, conn_loss, orth_loss, etp_loss, row_sum_loss)
    
    # def _log_sankey(self, P, step, tag="sankey"):
    #     """
    #     P: torch.Tensor [n_bits, n_pins]
    #     """
    #     P = P.detach().cpu().numpy()
    #     n_bits, n_pins = P.shape

    #     labels = (
    #         [f"b{i}" for i in range(n_bits)] +
    #         [f"p{j}" for j in range(n_pins)]
    #     )

    #     src, tgt, val = [], [], []

    #     for i in range(n_bits):
    #         for j in range(n_pins):
    #             if P[i, j] > 1e-3:
    #                 src.append(i)
    #                 tgt.append(n_bits + j)
    #                 val.append(float(P[i, j]))

    #     fig = go.Figure(data=[go.Sankey(
    #         node=dict(label=labels, pad=10),
    #         link=dict(source=src, target=tgt, value=val)
    #     )])

    #     path = os.path.join(self.conn_log_dir, f"{tag}epoch{step}.html")
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     fig.write_html(path)

    def _extract_frame(self, epoch, P, threshold = 1e-3):
        P = P.detach().cpu().numpy()
        n_bits, n_pins = P.shape
        src, tgt, val = [], [], []

        for b in range(n_bits):
            for p in range(n_pins):
                if P[b, p] != 0:
                    src.append(b)
                    tgt.append(n_bits + p)
                    val.append(float(P[b, p]))

        return dict(
            epoch=epoch,
            src=src,
            tgt=tgt,
            val=val,
            n_bits=n_bits,
            n_pins=n_pins
        )

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
                    P_soft = gumbel_sinkhorn(
                        logit_col,
                        tau=self.temp,
                        n_iters=10
                    )[0]
                    max_idx = torch.argmax(P_soft, dim=0)
                    f.write(f"Col {j} : ")
                    f.write(f"{max_idx.numpy().astype(int)}\n")
                    # plot sankey
                    if j in [14, 15, 16, 17]:
                        # P_hard = torch.zeros_like(logit_col)
                        # P_hard[max_idx, range(logit_col.shape[1])] = 1
                        # P_soft = F.softmax(logit_col / self.temp, dim=0)
                        
                        # self._log_sankey(P_soft, epoch, tag=f"sankey/soft/stage{i}_col{j}_")
                        # self._log_sankey(P_soft, epoch, tag=f"sankey/hard/stage{i}_col{j}_")
                        key = (i, j)
                        self.sankey_frames_soft[key].append(self._extract_frame(epoch, P_soft))
                        # self.sankey_frames_hard[key].append(self._extract_frame(epoch, P_hard))

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
    
    def _export_sankey(self, tag="sankey"):

        def export_group(frames, name):
            if not frames:
                return

            n_bits = frames[0]["n_bits"]
            n_pins = frames[0]["n_pins"]

            x = [0.0] * n_bits + [1.0] * n_pins
            y = (
                [0.05 + 0.9 * i / (n_bits - 1) if n_bits > 1 else 0.5 for i in range(n_bits)] +
                [0.05 + 0.9 * i / (n_pins - 1) if n_pins > 1 else 0.5 for i in range(n_pins)]
            )

            labels = (
                [f"b{i}" for i in range(n_bits)] +
                [f"p{j}" for j in range(n_pins)]
            )

            plotly_frames = []
            steps = []

            for k, fr in enumerate(frames):
                plotly_frames.append(go.Frame(
                    data=[go.Sankey(
                        arrangement="fixed",
                        node=dict(
                            label=labels,
                            x=x,
                            y=y,
                            pad=10
                        ),
                        link=dict(
                            source=fr["src"],
                            target=fr["tgt"],
                            value=fr["val"]
                        )
                    )],
                    name=str(fr["epoch"])
                ))

                steps.append(dict(
                    method="animate",
                    args=[[str(fr["epoch"])],
                        {"mode": "immediate", "frame": {"duration": 300, "redraw": False}, "transition": {"duration": 0}}],
                    label=str(fr["epoch"])
                ))

            fig = go.Figure(
                data=plotly_frames[0].data,
                frames=plotly_frames,
                layout=go.Layout(
                    title=name,
                    updatemenus=[dict(
                        type="buttons",
                        showactive=False,
                        buttons=[dict(label="Play", method="animate", args=[None])]
                    )],
                    sliders=[dict(steps=steps)]
                )
            )

            fig.write_html(os.path.join(self.conn_log_dir, f"{name}.html"))

        # export all
        for (stage, col), frames in self.sankey_frames_soft.items():
            export_group(frames, f"soft_stage{stage}_col{col}")

        # for (stage, col), frames in self.sankey_frames_hard.items():
            # export_group( frames, f"hard_stage{stage}_col{col}")
    
    def _save_logits(self, path):
        cpu_copy = []
        for stage in self.logits:
            stage_copy = []
            for p in stage:
                if p is None:
                    stage_copy.append(None)
                else:
                    stage_copy.append(p.detach().cpu())
            cpu_copy.append(stage_copy)

        torch.save(cpu_copy, path)
    
    def _load_logits(self, path, device):
        saved = torch.load(path, map_location=device)
        for s in range(self.N_STAGE):
            for j in range(self.N_COL):
                if saved[s][j] is None:
                    self.logits[s][j] = None
                else:
                    self.logits[s][j] = saved[s][j].to(device)
                    self.logits[s][j].requires_grad_(True)


    def train(self):
        # init DataLoader
        loader = IP_generator(self.n_bits, batch=self.batch_size, exhaustive=False)
        eval_loader = IP_generator(self.n_bits, batch=self.batch_size*5, exhaustive=False)
        print(f"[START TRAINING]")
        self.best_ckpt = 0
        best_eval_loss = float("inf")
        best_eval_loss_diff = float("inf")
        self.best_ckpt_path = os.path.join(self.conn_log_dir, "best.pt")
        from collections import defaultdict
        self.sankey_frames_hard = defaultdict(list)
        self.sankey_frames_soft = defaultdict(list)
        self.lambda_orth = 1
        for epoch in range(self.num_epochs):
            # get input pattern batch [B, N_COL, N_PIN]
            input_batch = next(loader)

            # self._update_tau(epoch)
            # self._update_lambda(epoch)
            loss, conn_loss, orth_loss, etp_loss, row_sum_loss = self._compute_loss(input_batch, Train=True, Baseline=False)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self._update_schedules(epoch, entropy=etp_loss)
            if epoch < self.warmup_epochs:
                warmup_lr = self.lr * (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            else:
                self.scheduler.step()
            
            if epoch > 500:
                self.eval_period = 1
                self.log_period = 1

            # self.writer.add_scalar("loss/train", loss, epoch)
            self.writer.add_scalar("loss/orth", orth_loss, epoch)
            self.writer.add_scalar("loss/conn", conn_loss, epoch)
            self.writer.add_scalar("loss/entropy", etp_loss, epoch)
            self.writer.add_scalar("loss/row_sum", row_sum_loss, epoch)
            # print
            if epoch % self.print_period == 0 or epoch == self.num_epochs - 1:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"[TRAIN {epoch}] Train Loss={loss.item():.4f}, LR={current_lr:.6f}, Tau={self.temp:.4f}, lambda_orth={self.lambda_orth:.2f}, orth_loss={orth_loss:.2f}, row_sum_loss = {row_sum_loss:.2f}, conn_loss={(conn_loss).item():.4f}, entropy={etp_loss.item():.2f}")

            # record connection
            if epoch % self.log_period == 0 or epoch == self.num_epochs - 1:
                self._log_conn(epoch)

            # EVAL
            if epoch % self.eval_period == 0 or epoch == self.num_epochs - 1:
                # Record eval diff and random conn diff
                with open(self.eval_log, "a") as f:
                    f.write(f"\n-- Epoch {epoch} --\n")
                with torch.no_grad():
                    input_batch = next(eval_loader)
                    eval_loss, _, _, _, _ = self._compute_loss(input_batch, Train=False, Baseline=False, Log=True)
                    rand_loss, _, _, _, _ = self._compute_loss(input_batch, Train=False, Baseline=True, Log=True)
                # self.writer.add_scalar("loss/eval_loss", eval_loss, epoch)
                self.writer.add_scalars(
                    "loss/Eval_vs_Random",
                    {
                        "eval_loss": eval_loss,
                        "rand_loss": rand_loss,
                    },
                    global_step=epoch,
                )
                print(f"##################[Eval {epoch}] Eval Loss={eval_loss.item():.6f}, Random Conn Loss={rand_loss.item():.6f}###################")
                # record train loss and eval loss
                with open(self.train_loss_csv, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, loss.item(), eval_loss.item()])
                if eval_loss.item() - rand_loss.item() < best_eval_loss_diff:
                    best_eval_loss = eval_loss.item()
                    best_eval_loss_diff = eval_loss.item() - rand_loss.item()
                    self.best_ckpt = epoch
                    self._save_logits(self.best_ckpt_path)
                    print(f"[BEST UPDATED] Epoch {epoch}, Eval Loss = {best_eval_loss:.2f}, Best Eval Loss - Rand Loss = {best_eval_loss_diff:.2f}")
            else:
                with open(self.train_loss_csv, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, loss.item(), None])
        
        print(f"[EXPORT SANKEY]")
        self._export_sankey()

    def test(self):
        self._load_logits(self.best_ckpt_path, device=device)
        print(f"Loaded best model from epoch {self.best_ckpt}")
        loader = IP_generator(self.n_bits, batch=int(self.test_batch_size), exhaustive=False)
        print(f"[START TESTING]")

        # get input pattern batch [B, N_COL, N_PIN]
        input_batch = next(loader)
    
        with torch.no_grad():
            eval_loss, _, _, _, _ = self._compute_loss(input_batch, Train=False, Baseline=False, Log=True, log_path=self.final_log)
            rand_loss, _, _, _, _ = self._compute_loss(input_batch, Train=False, Baseline=True, Log=True, log_path=self.final_log)

        print(f"[TEST] Test Loss={eval_loss.item():.6f}, Random Conn Loss={rand_loss.item():.6f}")

