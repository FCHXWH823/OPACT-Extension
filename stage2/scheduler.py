import math

class ThreePhaseScheduler:
    def __init__(self, cfg, num_epochs):

        # ===== phase boundaries =====
        self.t0 = cfg.get("t0", 0)
        self.t1 = cfg.get("t1", num_epochs - 1)

        # ===== phase values =====
        self.start = cfg.get("start", 1.0)   # value at epoch 0
        self.v0 = cfg.get("v0", 1.0)         # value at t0
        self.v1 = cfg.get("v1", 1.0)         # value at t1
        self.end = cfg.get("end", 1.0)       # final value

        # ===== phase types =====
        self.type0 = cfg.get("type0", "linear")
        self.type1 = cfg.get("type1", "linear")
        self.type2 = cfg.get("type2", "constant")

        # ===== entropy trigger =====
        self.entropy_th = cfg.get("entropy_threshold", None)

        self.num_epochs = num_epochs

    # ------------------------------------------------------------
    # interpolation kernels
    # ------------------------------------------------------------
    def _interp(self, v_start, v_end, progress, mode):
    
        progress = min(max(progress, 0.0), 1.0)

        if mode == "linear":
            return v_start + (v_end - v_start) * progress

        elif mode == "exp":
            if v_start <= 0:
                v_start = 1
            return v_start * (v_end / v_start) ** progress

        elif mode == "cosine":
            return v_end + 0.5 * (v_start - v_end) * (1 + math.cos(math.pi * progress))

        elif mode == "constant":
            return v_start

        else:
            raise ValueError(f"Unknown scheduler mode: {mode}")

    # ------------------------------------------------------------
    # main API
    # ------------------------------------------------------------
    def get(self, epoch, entropy=None):
        """
        Args:
            epoch: current epoch
            entropy: optional entropy value for trigger
        """

        # ===== entropy trigger: jump to phase1 =====
        if epoch < self.t0 and self.entropy_th is not None and entropy is not None:
            if entropy < self.entropy_th:
                self.t0=epoch

        # ===== phase0 =====
        if epoch < self.t0:
            progress = epoch / max(1, self.t0)
            self.start
            return self._interp(self.start, self.v0, progress, self.type0)

        # ===== phase1 =====
        if epoch < self.t1:
            progress = (epoch - self.t0) / max(1, self.t1 - self.t0)
            return self._interp(self.v0, self.v1, progress, self.type1)

        # ===== phase2 =====
        progress = (epoch - self.t1) / max(1, self.num_epochs - self.t1)
        return self._interp(self.v1, self.end, progress, self.type2)