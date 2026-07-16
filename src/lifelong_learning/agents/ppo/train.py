from __future__ import annotations

import copy
import os
import warnings

# [FIX] Silence TensorFlow OneDNN warning (must be before torch/tensorflow imports)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# [FIX] Silence pkg_resources deprecation warning from pygame
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import time
import numpy as np
import torch
import gymnasium as gym
from dataclasses import dataclass, field
from collections import deque

from lifelong_learning.agents.ppo.ppo import PPOConfig, ppo_update
from lifelong_learning.agents.ppo.network import CNNActorCritic
from lifelong_learning.agents.ppo.world_model import SimpleWorldModel
from lifelong_learning.agents.ppo.buffers import RolloutBuffer
from lifelong_learning.agents.ppo.episodic_memory import EpisodicMemory
from lifelong_learning.utils.seeding import seed_everything
from lifelong_learning.utils.logger import DataLogger
from lifelong_learning.envs.make_env import make_env


def configure_runtime_threads(num_threads: int | None):
    """Cap per-process CPU thread pools to avoid async worker oversubscription."""
    if num_threads is None:
        return

    thread_count = max(1, int(num_threads))
    thread_count_str = str(thread_count)
    for env_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[env_var] = thread_count_str

    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(thread_count)
    except RuntimeError:
        # PyTorch only allows this before inter-op work starts; subsequent resets can skip it.
        pass


# LOOP-0007 code-free two-timescale mechanisms: param groups whose LR *tracks* the
# main LR scaled by a cfg lever, keyed by group name -> cfg attribute holding the
# scale. The Brain's proven LR lever still moves these groups (design choice (b)),
# just damped, instead of being silently bypassed. The "decoder" group
# (trainable-neuromod) is deliberately absent here — it holds a fixed LR.
_SCALED_LR_GROUPS = {
    "critic": "critic_lr_scale",   # cand 1: damp post-switch critic whiplash
    "encoder": "encoder_lr_scale",  # cand 5: slow shared encoder vs fast heads
}


def apply_inner_lr(optimizer, lr: float, cfg=None) -> None:
    """Set the main param-group LR and mirror any scaled two-timescale group.

    Group 0 ("main") always carries the LR the Brain controls (action[0]) and the
    anneal schedule — the proven-causal scalar lever. LOOP-0007 candidates 1/5
    (research note 0004) carve the critic head / shared encoder into their own
    groups whose LR tracks the main LR scaled by a cfg lever (see
    _SCALED_LR_GROUPS). This helper is the single write path for the main LR — call
    it everywhere the main LR is set (anneal, Brain lever, eval Brain lever) so the
    scaled groups stay in sync. When no lever is active it is a plain
    param_groups[0] write, byte-identical to the previous behaviour.
    """
    optimizer.param_groups[0]["lr"] = lr
    if cfg is None:
        return
    for group in optimizer.param_groups:
        attr = _SCALED_LR_GROUPS.get(group.get("name"))
        if attr is None:
            continue
        scale = getattr(cfg, attr, 1.0)
        if group.get("name") == "critic" and getattr(cfg, "critic_lr_oracle_scale", 0.0) > 0.0:
            # O1 oracle-timed critic damp (research note 0005): while the post-switch
            # window armed by a DETECTED regime switch is live, damp the critic group;
            # otherwise the group tracks the main LR exactly (scale is 1.0 in the pure
            # oracle condition, but the damp composes with a static critic_lr_scale too).
            if getattr(cfg, "critic_oracle_remaining", 0) > 0:
                scale = scale * cfg.critic_lr_oracle_scale
            group["lr"] = lr * scale
            continue
        if scale != 1.0:
            group["lr"] = lr * scale


def resolve_inner_save_dir(*, save_dir: str, log_dir: str, logger_full_dir: str) -> str:
    """
    Keep inner checkpoints colocated with the concrete logger directory.

    MetaEnv launches multiple async outer workers that share an episode-level
    parent folder. When the caller points `save_dir` at `<episode>/inner_checkpoints`,
    redirect checkpoints into each worker's actual logger directory instead so
    workers do not contend over the same checkpoint folder.
    """
    normalized_parent = os.path.normpath(os.path.dirname(save_dir))
    normalized_log_dir = os.path.normpath(log_dir)
    if (
        os.path.basename(save_dir) == "inner_checkpoints"
        and normalized_parent == normalized_log_dir
    ):
        return os.path.join(logger_full_dir, "inner_checkpoints")
    return save_dir


# =========================================================================
# Inner Training State (exposed for Meta-RL "Brain" controller)
# =========================================================================

@dataclass
class InnerTrainState:
    """
    Mutable state of the inner Dyna-PPO training loop.

    Exposed so the Brain meta-agent can:
      1. Read training signals (stats, counters)
      2. Write hyperparameter adjustments (lr, ent_coef, intrinsic_coef, imagined_horizon)
    """
    # --- Models & optimizers ---
    model: CNNActorCritic = field(repr=False)
    anchor_model: CNNActorCritic | None = field(repr=False)
    optimizer: torch.optim.Adam = field(repr=False)
    world_model: SimpleWorldModel = field(repr=False)
    wm_optimizer: torch.optim.Adam = field(repr=False)
    buffer: RolloutBuffer = field(repr=False)

    # --- Environments ---
    envs: gym.vector.VectorEnv = field(repr=False)
    obs_shape: tuple = field(default=())
    n_actions: int = 0
    num_envs: int = 16

    # --- Config (mutable hyperparams the Brain can adjust) ---
    cfg: PPOConfig = field(default_factory=PPOConfig)
    intrinsic_coef: float = 0.1
    intrinsic_reward_clip: float = 0.1
    imagined_horizon: int = 5
    replay_ratio: float = 0.0
    replay_prioritization: float = 0.0

    # --- Episodic Memory ---
    episodic_memory: EpisodicMemory = field(default=None, repr=False)
    episodic_memory_capacity: int = 50000
    anneal_lr: bool = True

    # --- Counters ---
    global_step: int = 0
    current_update: int = 0
    num_updates: int = 0
    start_time: float = field(default_factory=time.time)

    # --- Live tensors ---
    obs_t: torch.Tensor = field(default=None, repr=False)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # --- Per-env episode trackers ---
    running_returns: np.ndarray = field(default=None, repr=False)
    running_lengths: np.ndarray = field(default=None, repr=False)

    # --- Outcome tracking ---
    outcome_window: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_outcome_window: deque = field(default_factory=lambda: deque(maxlen=20))

    # --- Logger ---
    logger: DataLogger = field(default=None, repr=False)
    run_name: str = ""

    # --- Checkpointing ---
    save_dir: str = "checkpoints"
    save_every_updates: int = 50

    # --- Regime switching ---
    regime_step_counter: list = field(default_factory=lambda: [0])
    current_mode_regime: int = 0

    # --- LOOP-0007 cand 4: surprise-triggered exploration spike (research note 0004) ---
    # A generic change-point detector on the inner agent's own per-update TD-error surprise
    # (value_loss) — NOT the Brain code. When value_loss jumps > threshold above its EMA, a
    # regime switch is inferred and ent_coef/intrinsic_coef are transiently boosted for a few
    # updates, faster than the Brain's decision_interval cadence. threshold 0.0 == off.
    surprise_spike_threshold: float = 0.0
    surprise_value_ema: float | None = None
    surprise_spike_remaining: int = 0

    # --- LOOP-0007 cand 2: ReDo dormant-neuron reset (research note 0004; Sokar 2023) ---
    # Every redo_interval updates, reset dormant hidden units in the actor/critic heads to
    # restore plasticity; the dormant-fraction probe is logged. 0 == off.
    redo_interval: int = 0

    # --- O2 oracle rung (research note 0005): policy-swap headroom topline ---
    # DIAGNOSTIC ceiling, never a method: bank the full learner (model, world model,
    # optimizer states) per regime at switch-away and restore it on regime revisit —
    # measures how much post-switch score a zero-forgetting agent could achieve on
    # this instrument. False == off (baseline-identical).
    policy_swap_topline: bool = False
    policy_swap_snapshots: dict = field(default_factory=dict, repr=False)
    # G-DECOMP swap-scope ladder (action tree 2026-07-09 ★): narrows what the O2 swap
    # banks/restores so the +0.249 ceiling can be attributed. "full" == the original O2.
    policy_swap_scope: str = "full"

    # --- G3 head-bank memory (W2A; routed by note 0012, LOOP-0012) ---
    # K weight slots for the actor/critic heads with pluggable trigger (oracle switch /
    # value-loss surprise) and selection (oracle regime id / other-at-K=2 / banked-critic
    # value error). The METHOD version of the heads-scope swap: 0 slots == off.
    head_bank_slots: int = 0
    head_bank_trigger: str = "oracle"
    head_bank_select: str = "oracle"
    head_bank_surprise_threshold: float = 1.0
    head_bank: dict = field(default_factory=dict, repr=False)
    head_bank_active: int = 0
    head_bank_value_ema: float | None = None
    head_bank_cooldown: int = 0
    # Scratch WM module for drift-robust fingerprint scoring (reward_fp): banked WMs are
    # loaded here for evaluation so the live world model is never touched by selection.
    head_bank_scratch_wm: object = field(default=None, repr=False)
    # --- per-step trigger (LOOP-0014, log 0012): success-collapse detector on episode
    # terminations, model-free. At a goal swap the per-episode success rate falls
    # ~0.9 -> ~0 within a handful of terminal events (16 envs terminate every ~3 steps),
    # while WM reward-prediction errors proved too noisy per event (calibration
    # 2026-07-15: abs bar precision 0.04; relative bar 0.03 with recall loss — the live
    # reward head is chronically underconfident at goal events). Fires mid-rollout when
    # the fast success EMA collapses below threshold x slow EMA (oracle-swap precedent
    # for mid-rollout weight loads); the slow EMA follows sustained change, so a
    # first-exposure regime (nothing good banked) stops re-firing on its own.
    head_bank_step_threshold: float = 0.25   # fire when ema_fast < thr * ema_slow
    # Half the first regime: every learning-phase wobble fire in calibration sat below
    # ~45k steps (shadow runs 2026-07-16); a switch can't be *restored* before anything
    # was banked anyway, so the warmup costs nothing.
    head_bank_step_warmup: int = 50000
    head_bank_step_cooldown: int = 2048      # min steps between fires
    head_bank_step_shadow: bool = False      # log fires, never switch (calibration mode)
    head_bank_step_ema_fast: float | None = None
    head_bank_step_ema_slow: float | None = None
    head_bank_step_events: int = 0           # terminal events since last fire/reset
    head_bank_step_cooldown_until: int = 0


# Step-trigger shape (fixed; only the collapse threshold is exposed as a lever).
STEP_FAST_DECAY = 0.8           # fast success EMA (per terminal event)
STEP_SLOW_DECAY = 0.995         # slow success EMA (the regime baseline)
STEP_MIN_SLOW = 0.5             # only fire out of a *learned* regime
STEP_MIN_EVENTS = 8             # terminal events needed since last fire/reset

# Cand-4 spike shape (fixed; only the trigger threshold is exposed as a lever).
SURPRISE_SPIKE_FACTOR = 2.0     # transient multiplier on ent_coef + intrinsic_coef
SURPRISE_SPIKE_UPDATES = 3      # updates the boost persists after a detected switch
SURPRISE_EMA_DECAY = 0.9        # EMA smoothing of the value-loss baseline


def _update_surprise_spike(state, value_loss_now: float) -> None:
    """Cand-4 change-point detector (research note 0004): arm the exploration spike when the
    per-update TD-error surprise (value_loss) jumps more than surprise_spike_threshold above
    its EMA baseline — a generic switch signal from the agent's own learning dynamics, not the
    Brain code. Mutates state.surprise_spike_remaining / surprise_value_ema in place. No-op when
    the mechanism is off. First observation only seeds the baseline (never triggers)."""
    s = state
    if s.surprise_spike_threshold <= 0.0:
        return
    if s.surprise_value_ema is None:
        s.surprise_value_ema = value_loss_now
        return
    if value_loss_now > s.surprise_value_ema * (1.0 + s.surprise_spike_threshold):
        s.surprise_spike_remaining = SURPRISE_SPIKE_UPDATES
    s.surprise_value_ema = (
        SURPRISE_EMA_DECAY * s.surprise_value_ema
        + (1.0 - SURPRISE_EMA_DECAY) * value_loss_now
    )


# Cand-2 ReDo (Sokar 2023): dormancy threshold on the normalized per-neuron activation
# score. tau=0 resets only exactly-dead ReLU units; a small tau also catches near-dead.
REDO_TAU = 0.025


def _reset_adam_state(optimizer, param, rows=None, cols=None) -> None:
    """Zero the Adam moment estimates for reset neurons so they restart cleanly (ReDo)."""
    st = optimizer.state.get(param)
    if not st:
        return
    for key in ("exp_avg", "exp_avg_sq"):
        if key in st:
            if rows is not None:
                st[key][rows] = 0.0
            if cols is not None:
                st[key][:, cols] = 0.0


def redo_reset_heads(model, optimizer, obs, tau: float = REDO_TAU) -> dict:
    """LOOP-0007 cand 2 (ReDo, Sokar 2023): reset dormant hidden units in the actor & critic
    heads to restore plasticity. A unit is dormant if its normalized mean-abs activation over
    the batch is <= tau. Reset = re-init incoming weights (orthogonal) + zero bias, zero the
    outgoing weights (so the reset does not perturb the output immediately), and clear the Adam
    moments for the touched params. The trigger is a generic activation statistic, NOT the Brain
    code. Returns {head: dormant_fraction} — the registered mechanism probe (note 0004)."""
    activations: dict = {}

    def _hook(name):
        def _capture(module, inp, out):
            activations[name] = out.detach()
        return _capture

    handles = [
        model.actor_head[1].register_forward_hook(_hook("actor")),
        model.critic_head[1].register_forward_hook(_hook("critic")),
    ]
    try:
        with torch.no_grad():
            model(obs)
    finally:
        for h in handles:
            h.remove()

    fractions: dict = {}
    heads = [
        ("actor", model.actor_head[0], model.actor_head[2]),
        ("critic", model.critic_head[0], model.critic_head[2]),
    ]
    for name, in_layer, out_layer in heads:
        act = activations[name]  # (batch, hidden)
        mean_abs = act.abs().mean(dim=0)
        score = mean_abs / (mean_abs.mean() + 1e-9)
        dormant = (score <= tau).nonzero(as_tuple=True)[0]
        fractions[name] = float(dormant.numel()) / float(mean_abs.numel())
        if dormant.numel() == 0:
            continue
        with torch.no_grad():
            new_rows = torch.empty(
                dormant.numel(), in_layer.weight.shape[1], device=in_layer.weight.device
            )
            torch.nn.init.orthogonal_(new_rows, gain=np.sqrt(2))
            in_layer.weight[dormant] = new_rows
            if in_layer.bias is not None:
                in_layer.bias[dormant] = 0.0
            out_layer.weight[:, dormant] = 0.0
            _reset_adam_state(optimizer, in_layer.weight, rows=dormant)
            if in_layer.bias is not None:
                _reset_adam_state(optimizer, in_layer.bias, rows=dormant)
            _reset_adam_state(optimizer, out_layer.weight, cols=dormant)
    return fractions


# O1 oracle-timed critic damp (research note 0005): damp window length. The damp covers
# the remainder of the update in which the switch is detected plus this many subsequent
# updates. Fixed shape — only the damp scale is exposed as a flag (one variant, no
# multiplicity), mirroring the surprise-spike convention.
CRITIC_ORACLE_UPDATES = 15


# G-DECOMP (action tree 2026-07-09 ★): key-prefix filters for scoped snapshots.
# "heads+encoder" is the policy net's trainable trunk; feature_norm (plasticity_norm,
# off in the paper preset) and the aux code head (aux_code_coef=0 there) sit outside it.
SWAP_SCOPE_PREFIXES = {
    "heads": ("actor_head.", "critic_head."),
    "heads+encoder": ("actor_head.", "critic_head.", "encoder."),
}
SWAP_SCOPES = ("heads", "heads+encoder", "world_model", "full")


def _snapshot_learner(state, scope: str = "full") -> dict:
    """O2 policy-swap topline (research note 0005): deep-copy everything the learner
    would 'forget' across a regime switch — model, world model, and both Adam states.

    G-DECOMP: `scope` narrows the bank so the ceiling decomposes — "heads" /
    "heads+encoder" (policy-net weights only), "world_model" (WM weights only), or
    "full" (the original O2). Partial scopes carry NO optimizer state, so full-vs-
    partial gaps also bound the optimizer-curvature share of the ceiling."""
    s = state
    if scope == "full":
        return {
            "model": copy.deepcopy(s.model.state_dict()),
            "world_model": copy.deepcopy(s.world_model.state_dict()),
            "optimizer": copy.deepcopy(s.optimizer.state_dict()),
            "wm_optimizer": copy.deepcopy(s.wm_optimizer.state_dict()),
        }
    if scope == "world_model":
        return {"world_model": copy.deepcopy(s.world_model.state_dict())}
    prefixes = SWAP_SCOPE_PREFIXES[scope]
    return {
        "model_partial": {
            k: copy.deepcopy(v)
            for k, v in s.model.state_dict().items()
            if k.startswith(prefixes)
        }
    }


def _restore_learner(state, snap: dict) -> None:
    """O2: restore a banked per-regime learner — only the pieces the snapshot carries.
    Optimizer state_dicts carry the LRs from snapshot time, so the current main LR
    (anneal / Brain lever) is re-applied after the load — the swap must never clobber
    the live LR schedule."""
    s = state
    if "model_partial" in snap:
        merged = s.model.state_dict()
        merged.update(snap["model_partial"])
        s.model.load_state_dict(merged)
    if "model" in snap:
        s.model.load_state_dict(snap["model"])
    if "world_model" in snap:
        s.world_model.load_state_dict(snap["world_model"])
    if "optimizer" in snap:
        main_lr = s.optimizer.param_groups[0]["lr"]
        s.optimizer.load_state_dict(copy.deepcopy(snap["optimizer"]))
        s.wm_optimizer.load_state_dict(copy.deepcopy(snap["wm_optimizer"]))
        apply_inner_lr(s.optimizer, main_lr, s.cfg)


# --- G3 head-bank memory (W2A; note 0012, LOOP-0012) -------------------------------
# The method form of the ladder's heads-scope swap: K persistent weight slots for the
# actor/critic heads. Trigger and selection are pluggable so the oracle ingredients can
# be replaced one at a time (the de-oracling ladder). Cooldown matches the calibrated
# detector (scripts/calibrate_surprise_trigger.py: thr 1.0 -> precision .84, recall .85).
HEAD_BANK_COOLDOWN_UPDATES = 8
HEAD_BANK_PREFIXES = SWAP_SCOPE_PREFIXES["heads"]
WM_REWARD_HEAD_PREFIX = ("reward_head.",)


def _head_bank_snapshot(state) -> dict:
    """A slot: the policy heads, plus — in reward_error mode only — the WM reward head
    (257 params) banked as the slot's regime fingerprint (note 0013: the regimes differ
    only in reward, so the reward head is the identity signal value fit lacked)."""
    s = state
    snap = {
        "policy": {
            k: copy.deepcopy(v)
            for k, v in s.model.state_dict().items()
            if k.startswith(HEAD_BANK_PREFIXES)
        }
    }
    if s.head_bank_select == "reward_error":
        snap["wm_reward"] = {
            k: copy.deepcopy(v)
            for k, v in s.world_model.state_dict().items()
            if k.startswith(WM_REWARD_HEAD_PREFIX)
        }
    if s.head_bank_select == "reward_fp":
        # Drift-robust fingerprint (LOOP-0015): bank the ENTIRE world model so the slot's
        # regime signature is scored in its own frozen feature space (~1.5M params/slot,
        # scoring-only — never loaded into the live learner).
        snap["wm_full"] = copy.deepcopy(s.world_model.state_dict())
    return snap


def _head_bank_load(state, snap: dict) -> None:
    s = state
    merged = s.model.state_dict()
    merged.update(snap["policy"])
    s.model.load_state_dict(merged)
    if "wm_reward" in snap:
        wm = s.world_model.state_dict()
        wm.update(snap["wm_reward"])
        s.world_model.load_state_dict(wm)


def _head_bank_select_value_error(state) -> int:
    """Content-addressable selection via critic value fit on the freshest rollout.
    (LOOP-0012's P-G3c arm ran WITHOUT the spawn-until-full allocation below, so this
    selector was never actually exercised there — re-run properly in LOOP-0013.)"""
    s = state
    candidates = sorted(set(s.head_bank) | {s.head_bank_active})
    if len(candidates) == 1:
        return s.head_bank_active
    obs = s.buffer.obs.reshape((-1,) + s.obs_shape)
    returns = s.buffer.returns.reshape(-1)
    live = _head_bank_snapshot(s)
    best_slot, best_err = s.head_bank_active, None
    with torch.no_grad():
        for slot in candidates:
            snap = live if slot == s.head_bank_active else s.head_bank[slot]
            _head_bank_load(s, snap)
            _, value = s.model.forward(obs)
            err = torch.mean((value.reshape(-1) - returns) ** 2).item()
            if best_err is None or err < best_err:
                best_slot, best_err = slot, err
    _head_bank_load(s, live)
    return best_slot


def _head_bank_select_reward_error(state) -> int:
    """Content-addressable selection via the regime fingerprint (pre-reg log 0010):
    score each slot's banked WM reward head on the freshest (obs, action) -> reward
    transitions and pick the argmin. Scored through the live WM trunk — trunk drift
    between banking and scoring is a registered risk. Falls back to the active slot
    when nothing else is banked."""
    s = state
    candidates = sorted(set(s.head_bank) | {s.head_bank_active})
    if len(candidates) == 1:
        return s.head_bank_active
    obs = s.buffer.obs.reshape((-1,) + s.obs_shape)
    actions = s.buffer.actions.reshape(-1).long()
    rewards = s.buffer.extrinsic_rewards.reshape(-1)
    live = {
        k: copy.deepcopy(v)
        for k, v in s.world_model.state_dict().items()
        if k.startswith(WM_REWARD_HEAD_PREFIX)
    }

    def _load_wm_head(head: dict) -> None:
        wm = s.world_model.state_dict()
        wm.update(head)
        s.world_model.load_state_dict(wm)

    best_slot, best_err = s.head_bank_active, None
    with torch.no_grad():
        for slot in candidates:
            head = live if slot == s.head_bank_active else s.head_bank[slot].get("wm_reward")
            if head is None:
                continue
            _load_wm_head(head)
            _, pred_reward = s.world_model.forward(obs, actions)
            err = torch.mean((pred_reward - rewards) ** 2).item()
            if best_err is None or err < best_err:
                best_slot, best_err = slot, err
    _load_wm_head(live)
    return best_slot


def _head_bank_select_reward_fp(state) -> int:
    """Drift-robust content addressing (LOOP-0015, log 0013): each slot's fingerprint is
    its entire banked world model, scored in its OWN frozen feature space on the freshest
    (obs, action) -> reward transitions — the fix for the LOOP-0013 drift null, where
    banked heads scored through the live trunk always lost to the fresh head. The ACTIVE
    slot is scored by the live WM ('no switch happened' is exactly what the live model
    represents), so at K=2 the selector is a fire verifier: flip on true switches, stay
    on false fires (the LOOP-0014 churn suppressor). Banked WMs are classifiers only —
    restores load policy heads only."""
    s = state
    candidates = sorted(set(s.head_bank) | {s.head_bank_active})
    if len(candidates) == 1:
        return s.head_bank_active
    obs = s.buffer.obs.reshape((-1,) + s.obs_shape)
    actions = s.buffer.actions.reshape(-1).long()
    rewards = s.buffer.extrinsic_rewards.reshape(-1)
    if s.head_bank_scratch_wm is None:
        s.head_bank_scratch_wm = copy.deepcopy(s.world_model)
    best_slot, best_err = s.head_bank_active, None
    with torch.no_grad():
        for slot in candidates:
            if slot == s.head_bank_active:
                model = s.world_model
            else:
                fp = s.head_bank[slot].get("wm_full")
                if fp is None:
                    continue
                s.head_bank_scratch_wm.load_state_dict(fp)
                model = s.head_bank_scratch_wm
            _, pred_reward = model.forward(obs, actions)
            err = torch.mean((pred_reward - rewards) ** 2).item()
            if best_err is None or err < best_err:
                best_slot, best_err = slot, err
    return best_slot


def _head_bank_switch(state, target_slot: int) -> None:
    """Bank the active slot's heads; load the target slot's if previously banked (else the
    current heads keep running — a warm start for a never-seen slot)."""
    s = state
    s.head_bank[s.head_bank_active] = _head_bank_snapshot(s)
    restored = False
    if target_slot != s.head_bank_active:
        snap = s.head_bank.get(target_slot)
        if snap is not None:
            _head_bank_load(s, snap)
            restored = True
        s.head_bank_active = target_slot
    s.logger.scalar("brain_neuromod/head_bank_active", float(s.head_bank_active), s.global_step)
    s.logger.scalar("brain_neuromod/head_bank_restored", float(restored), s.global_step)


def _head_bank_surprise_check(state, value_loss_now: float) -> None:
    """A-R1 learned trigger: an independent EMA change-point detector on the per-update
    value loss (same math as cand-4's, own state — the exploration spike stays decoupled).
    On a detected spike: 'other' selection flips the K=2 slot; 'value_error' selection is
    content-addressable. First observation seeds the baseline; cooldown prevents thrash."""
    s = state
    if s.head_bank_value_ema is None:
        s.head_bank_value_ema = value_loss_now
        return
    if s.head_bank_cooldown > 0:
        s.head_bank_cooldown -= 1
    elif value_loss_now > s.head_bank_value_ema * (1.0 + s.head_bank_surprise_threshold):
        s.logger.scalar("brain_neuromod/head_bank_trigger_fired", 1.0, s.global_step)
        if s.head_bank_select == "other":
            target = 1 - s.head_bank_active
        else:
            # Spawn-until-full allocation (registered, log 0010): content addressing can
            # only choose among banked fingerprints — without this, K=2 never populates
            # slot 1 and the selector degenerates to bank-without-restore (the LOOP-0012
            # P-G3c artifact). First fires allocate fresh slots; addressing starts once
            # every slot is live.
            unused = [k for k in range(s.head_bank_slots)
                      if k != s.head_bank_active and k not in s.head_bank]
            if unused:
                target = unused[0]
            elif s.head_bank_select == "reward_fp":
                target = _head_bank_select_reward_fp(s)
            elif s.head_bank_select == "reward_error":
                target = _head_bank_select_reward_error(s)
            else:  # value_error
                target = _head_bank_select_value_error(s)
        _head_bank_switch(s, target)
        s.head_bank_cooldown = HEAD_BANK_COOLDOWN_UPDATES
    s.head_bank_value_ema = (
        SURPRISE_EMA_DECAY * s.head_bank_value_ema
        + (1.0 - SURPRISE_EMA_DECAY) * value_loss_now
    )


def _head_bank_step_check(state, reward, done) -> None:
    """Per-step trigger (LOOP-0014, log 0012): model-free success-collapse detector.
    Consumes episode terminations from the vector env (outcome = terminal reward > 0.05)
    and maintains fast/slow success EMAs; fires mid-rollout when the fast EMA collapses
    below head_bank_step_threshold x the slow EMA — the signature of a goal swap under a
    competent policy (~0.9 -> ~0 within a handful of terminal events). Gates: the slow
    EMA must show a learned regime (>= 0.5), >= 8 events since the last fire/reset,
    warmup, and a step cooldown. On fire the fast EMA resets optimistically to the slow
    EMA: a correct restore recovers success and stays quiet; a wrong flip re-collapses
    and self-corrects after the cooldown. The slow EMA keeps following sustained change,
    so a first-exposure regime (nothing good banked yet) stops re-firing on its own.
    Shadow mode logs would-be fires without switching (calibration)."""
    s = state
    if s.head_bank_slots <= 0 or s.head_bank_trigger != "step_surprise":
        return
    for i in np.flatnonzero(done):
        outcome = 1.0 if float(reward[i]) > 0.05 else 0.0
        if s.head_bank_step_ema_slow is None:
            s.head_bank_step_ema_slow = outcome
            s.head_bank_step_ema_fast = outcome
            continue
        s.head_bank_step_ema_slow = (
            STEP_SLOW_DECAY * s.head_bank_step_ema_slow + (1.0 - STEP_SLOW_DECAY) * outcome
        )
        s.head_bank_step_ema_fast = (
            STEP_FAST_DECAY * s.head_bank_step_ema_fast + (1.0 - STEP_FAST_DECAY) * outcome
        )
        s.head_bank_step_events += 1
        if (s.head_bank_step_ema_slow >= STEP_MIN_SLOW
                and s.head_bank_step_events >= STEP_MIN_EVENTS
                and s.head_bank_step_ema_fast < s.head_bank_step_threshold * s.head_bank_step_ema_slow
                and s.global_step >= s.head_bank_step_warmup
                and s.global_step >= s.head_bank_step_cooldown_until):
            s.logger.scalar("brain_neuromod/head_bank_step_fire", 1.0, s.global_step)
            s.head_bank_step_cooldown_until = s.global_step + s.head_bank_step_cooldown
            s.head_bank_step_events = 0
            s.head_bank_step_ema_fast = s.head_bank_step_ema_slow
            if not s.head_bank_step_shadow:
                _head_bank_switch(s, (s.head_bank_active + 1) % 2)
            return


def init_inner_training(
    env_id: str,
    cfg: PPOConfig,
    *,
    steps_per_regime: int | None = None,
    episodes_per_regime: int | None = None,
    start_regime: int = 0,
    num_regimes: int = 2,
    run_name: str | None = None,
    save_dir: str = "checkpoints",
    save_every_updates: int = 50,
    anneal_lr: bool = True,
    resume_path: str | None = None,
    intrinsic_coef: float = 0.1,
    intrinsic_reward_clip: float = 0.1,
    imagined_horizon: int = 5,
    wm_lr: float = 1e-4,
    log_dir: str = "runs",
    episodic_memory_capacity: int = 50000,
    replay_ratio: float = 0.0,
    replay_prioritization: float = 0.0,
    cpu_threads: int | None = None,
    trainable_neuromod: bool = False,
    neuromod_decoder_lr: float | None = None,
    actor_only_neuromod: bool = False,
    neuromod_gain_alpha: float = 0.0,
    grad_gate_neuromod: bool = False,
    critic_code_neuromod: bool = False,
    aux_code_coef: float = 0.0,
    critic_lr_scale: float = 1.0,
    encoder_lr_scale: float = 1.0,
    plasticity_norm: bool = False,
    surprise_spike_threshold: float = 0.0,
    redo_interval: int = 0,
    critic_lr_oracle_scale: float = 0.0,
    policy_swap_topline: bool = False,
    policy_swap_scope: str = "full",
    head_bank_slots: int = 0,
    head_bank_trigger: str = "oracle",
    head_bank_select: str = "oracle",
    head_bank_surprise_threshold: float = 1.0,
    head_bank_step_threshold: float = 0.25,
    head_bank_step_shadow: bool = False,
) -> InnerTrainState:
    """
    Initialize all components of the Dyna-PPO inner training loop.

    Returns an InnerTrainState that can be driven step-by-step via
    run_inner_update(), or used internally by train_ppo().
    """

    # Clone the config so meta-controller mutations stay local to this inner run.
    cfg = copy.deepcopy(cfg)
    cfg.aux_code_coef = aux_code_coef
    # LOOP-0007 cands 1/5: read by apply_inner_lr to keep the critic / encoder
    # group LR at main_lr * scale wherever the main LR is written. 1.0 == off.
    cfg.critic_lr_scale = critic_lr_scale
    cfg.encoder_lr_scale = encoder_lr_scale
    # O1 oracle-timed critic damp (research note 0005): apply_inner_lr damps the critic
    # group by this scale while critic_oracle_remaining > 0 (armed on each DETECTED
    # regime switch — ground truth from the env, not the Brain code). 0.0 == off.
    cfg.critic_lr_oracle_scale = critic_lr_oracle_scale
    cfg.critic_oracle_remaining = 0
    # G-DECOMP: validate up front — a typo'd scope must fail at init, not at the
    # first regime switch hours into a ladder run.
    if policy_swap_scope not in SWAP_SCOPES:
        raise ValueError(
            f"policy_swap_scope must be one of {SWAP_SCOPES}, got {policy_swap_scope!r}"
        )
    # G3 head bank: validate the trigger×selection combo at init. Oracle selection needs
    # the regime id (only the oracle trigger has it); 'other' is the K=2 degenerate mode;
    # value-error selection needs post-trigger data, which only the surprise trigger
    # guarantees (it fires after an update on new-regime rollouts).
    if head_bank_slots:
        if head_bank_slots < 2:
            raise ValueError("head_bank_slots must be >= 2 (or 0 to disable)")
        if head_bank_trigger not in ("oracle", "surprise", "step_surprise"):
            raise ValueError(
                f"head_bank_trigger must be oracle|surprise|step_surprise, got {head_bank_trigger!r}"
            )
        if head_bank_select not in ("oracle", "other", "value_error", "reward_error", "reward_fp"):
            raise ValueError(
                "head_bank_select must be oracle|other|value_error|reward_error|reward_fp, "
                f"got {head_bank_select!r}"
            )
        if head_bank_trigger == "oracle" and head_bank_select != "oracle":
            raise ValueError("the oracle trigger pairs only with oracle selection")
        if head_bank_trigger == "surprise" and head_bank_select == "oracle":
            raise ValueError("the surprise trigger has no regime id — use other|value_error selection")
        if head_bank_trigger == "step_surprise" and head_bank_select != "other":
            raise ValueError(
                "step_surprise v1 pairs only with 'other' selection (content addressing is "
                "drift-broken pending banked scoring paths — note 0013)"
            )
        if head_bank_select == "other" and head_bank_slots != 2:
            raise ValueError("'other' selection is only defined at head_bank_slots=2")
        if policy_swap_topline:
            raise ValueError("head_bank and policy_swap_topline are mutually exclusive")

    seed_everything(cfg.seed)
    configure_runtime_threads(cpu_threads)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    num_envs = max(1, int(cfg.num_envs))

    # -----------------------------------------------------------------
    # Environment Setup
    # -----------------------------------------------------------------

    # Shared step counter so all sub-envs switch regime simultaneously
    regime_step_counter = [0]

    def make_thunk(i: int):
        def thunk():
            return make_env(
                env_id=env_id,
                seed=cfg.seed + i,
                steps_per_regime=steps_per_regime,
                episodes_per_regime=episodes_per_regime,
                start_regime=start_regime,
                num_regimes=num_regimes,
                record_stats=False,
                shared_step_counter=regime_step_counter,
            )
        return thunk

    envs = gym.vector.SyncVectorEnv([make_thunk(i) for i in range(num_envs)])
    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    # -----------------------------------------------------------------
    # Model & Optimizer Setup
    # -----------------------------------------------------------------

    model = CNNActorCritic(
        obs_shape,
        n_actions,
        trainable_neuromod=trainable_neuromod,
        actor_only_neuromod=actor_only_neuromod,
        neuromod_gain_alpha=neuromod_gain_alpha,
        grad_gate_neuromod=grad_gate_neuromod,
        critic_code_neuromod=critic_code_neuromod,
        aux_code_head=aux_code_coef > 0.0,
        plasticity_norm=plasticity_norm,
    ).to(device)
    # Param groups. Group 0 ("main") always holds the LR the Brain lever + anneal
    # drive (via apply_inner_lr). Optional extra groups carve out params that need
    # a different LR: the neuromod decoder (fixed LR, docs/multi_agent/0002) and —
    # LOOP-0007 cands 1/5 — the critic head (LR = main * critic_lr_scale) and the
    # shared encoder (LR = main * encoder_lr_scale), both tracked so the Brain's
    # proven LR lever still reaches them, damped. Groups are built only when their
    # mechanism is active, so a plain run stays a single-group optimizer
    # byte-identical to the previous behaviour.
    decouple_decoder = trainable_neuromod and neuromod_decoder_lr is not None
    # O1 also needs the critic in its own group (at scale 1.0 outside damp windows).
    decouple_critic = critic_lr_scale != 1.0 or critic_lr_oracle_scale > 0.0
    decouple_encoder = encoder_lr_scale != 1.0
    if decouple_decoder or decouple_critic or decouple_encoder:
        reserved_ids: set[int] = set()
        param_groups = []
        if decouple_decoder:
            decoder_params = list(model.neuromodulator.decoder.parameters())
            reserved_ids |= {id(p) for p in decoder_params}
        if decouple_critic:
            critic_params = list(model.critic_head.parameters())
            reserved_ids |= {id(p) for p in critic_params}
        if decouple_encoder:
            encoder_params = list(model.encoder.parameters())
            reserved_ids |= {id(p) for p in encoder_params}
        main_params = [p for p in model.parameters() if id(p) not in reserved_ids]
        param_groups.append({"params": main_params, "lr": cfg.lr, "name": "main"})
        if decouple_decoder:
            param_groups.append(
                {"params": decoder_params, "lr": neuromod_decoder_lr, "name": "decoder"}
            )
        if decouple_critic:
            param_groups.append(
                {"params": critic_params, "lr": cfg.lr * critic_lr_scale, "name": "critic"}
            )
        if decouple_encoder:
            param_groups.append(
                {"params": encoder_params, "lr": cfg.lr * encoder_lr_scale, "name": "encoder"}
            )
        optimizer = torch.optim.Adam(param_groups, eps=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    anchor_model = CNNActorCritic(
        obs_shape,
        n_actions,
        trainable_neuromod=trainable_neuromod,
        actor_only_neuromod=actor_only_neuromod,
        neuromod_gain_alpha=neuromod_gain_alpha,
        grad_gate_neuromod=grad_gate_neuromod,
        critic_code_neuromod=critic_code_neuromod,
        aux_code_head=aux_code_coef > 0.0,
        plasticity_norm=plasticity_norm,
    ).to(device)
    anchor_model.load_state_dict(model.state_dict())
    anchor_model.eval()

    world_model = SimpleWorldModel(obs_shape, n_actions).to(device)
    wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=wm_lr)
    buffer = RolloutBuffer(cfg.num_steps, num_envs, obs_shape, device)

    # -----------------------------------------------------------------
    # Resume from Checkpoint
    # -----------------------------------------------------------------

    start_global_step = 0
    if resume_path is not None and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("Optimizer state loaded.")
        else:
            print("WARNING: Optimizer state not found in checkpoint.")

        if "world_model_state_dict" in ckpt:
            world_model.load_state_dict(ckpt["world_model_state_dict"])
            print("World Model state loaded.")
        else:
            print("WARNING: World Model state not found in checkpoint.")

        if "wm_optimizer_state_dict" in ckpt:
            wm_optimizer.load_state_dict(ckpt["wm_optimizer_state_dict"])
            print("World Model Optimizer state loaded.")

        if "global_step" in ckpt:
            start_global_step = ckpt["global_step"]
            print(f"Resuming from global_step={start_global_step}")

    # -----------------------------------------------------------------
    # Logging & Tracking
    # -----------------------------------------------------------------

    if run_name is None:
        run_name = f"ppo_{env_id}_s{cfg.seed}"

    logger = DataLogger(run_name=run_name, log_dir=log_dir)
    save_dir = resolve_inner_save_dir(
        save_dir=save_dir,
        log_dir=log_dir,
        logger_full_dir=logger.full_dir,
    )
    os.makedirs(save_dir, exist_ok=True)

    obs, info = envs.reset(seed=cfg.seed)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    num_updates = cfg.total_timesteps // (num_envs * cfg.num_steps)
    global_step = start_global_step
    start_update = global_step // (num_envs * cfg.num_steps) + 1

    ep_memory = EpisodicMemory(
        capacity=episodic_memory_capacity,
        obs_shape=obs_shape,
        device=device,
    )

    return InnerTrainState(
        model=model,
        anchor_model=anchor_model,
        optimizer=optimizer,
        world_model=world_model,
        wm_optimizer=wm_optimizer,
        buffer=buffer,
        envs=envs,
        obs_shape=obs_shape,
        n_actions=n_actions,
        num_envs=num_envs,
        cfg=cfg,
        intrinsic_coef=intrinsic_coef,
        intrinsic_reward_clip=intrinsic_reward_clip,
        imagined_horizon=imagined_horizon,
        replay_ratio=replay_ratio,
        replay_prioritization=replay_prioritization,
        anneal_lr=anneal_lr,
        global_step=global_step,
        current_update=start_update,
        num_updates=num_updates,
        start_time=time.time(),
        obs_t=obs_t,
        device=device,
        running_returns=np.zeros(num_envs),
        running_lengths=np.zeros(num_envs, dtype=int),
        outcome_window=deque(maxlen=100),
        logger=logger,
        run_name=run_name,
        save_dir=save_dir,
        save_every_updates=save_every_updates,
        episodic_memory=ep_memory,
        episodic_memory_capacity=episodic_memory_capacity,
        regime_step_counter=regime_step_counter,
        current_mode_regime=start_regime,
        surprise_spike_threshold=surprise_spike_threshold,
        redo_interval=redo_interval,
        policy_swap_topline=policy_swap_topline,
        policy_swap_scope=policy_swap_scope,
        head_bank_slots=head_bank_slots,
        head_bank_trigger=head_bank_trigger,
        head_bank_select=head_bank_select,
        head_bank_surprise_threshold=head_bank_surprise_threshold,
        head_bank_step_threshold=head_bank_step_threshold,
        head_bank_step_shadow=head_bank_step_shadow,
    )


# =========================================================================
# Single Update Step (the core unit the Brain calls)
# =========================================================================

def run_inner_update(state: InnerTrainState) -> dict:
    """
    Run ONE Dyna-PPO update cycle (collect → train WM → dream → PPO update).

    Mutates `state` in place (global_step, current_update, obs_t, etc.)
    and returns a stats dict with all signals the Brain needs.

    Returns:
        dict with keys like:
          - "mean_episodic_return", "success_rate", "failure_rate"
          - "mean_surprise", "wm_loss_state", "wm_loss_reward"
          - "policy_entropy", "policy_loss", "value_loss"
          - "current_lr", "current_ent_coef", "current_intrinsic_coef"
          - "current_imagined_horizon", "global_step", "done"
    """

    s = state  # alias
    update = s.current_update

    if update > s.num_updates:
        return {"done": True, "global_step": s.global_step}

    # -----------------------------------------------------------------
    # LOOP-0007 cand 4: surprise-triggered exploration spike (transient).
    # If a prior update's TD-error jump armed the spike, boost ent_coef +
    # intrinsic_coef for THIS update only (base values are restored after the
    # update, below, so the Brain's own settings are never overwritten). No-op
    # unless the mechanism is armed (threshold > 0 and a switch was detected).
    # -----------------------------------------------------------------
    spike_active = s.surprise_spike_remaining > 0
    base_ent_coef = s.cfg.ent_coef
    base_intrinsic_coef = s.intrinsic_coef
    if spike_active:
        s.cfg.ent_coef = base_ent_coef * SURPRISE_SPIKE_FACTOR
        s.intrinsic_coef = base_intrinsic_coef * SURPRISE_SPIKE_FACTOR
        s.surprise_spike_remaining -= 1

    # -----------------------------------------------------------------
    # Learning rate annealing
    # -----------------------------------------------------------------
    if s.anneal_lr:
        frac = 1.0 - (update - 1.0) / s.num_updates
        lrnow = frac * s.cfg.lr
        apply_inner_lr(s.optimizer, lrnow, s.cfg)
    else:
        lrnow = s.optimizer.param_groups[0]["lr"]

    # -----------------------------------------------------------------
    # O1 oracle-timed critic damp (research note 0005): keep the critic group
    # damped while the window armed by a detected switch is live (explicit
    # resync covers the no-anneal path), consume one window update, and log
    # the probe. The arm itself happens in the Phase-A switch handler below,
    # which also resyncs immediately so the damp covers the detection update's
    # own PPO phases.
    # -----------------------------------------------------------------
    if s.cfg.critic_lr_oracle_scale > 0.0:
        apply_inner_lr(s.optimizer, lrnow, s.cfg)
        oracle_live = s.cfg.critic_oracle_remaining > 0
        s.logger.scalar("brain_neuromod/critic_oracle_active", float(oracle_live), s.global_step)
        if oracle_live:
            s.cfg.critic_oracle_remaining -= 1

    s.buffer.reset()

    # =====================================================================
    # Phase A: Collect real experience
    # =====================================================================

    episodic_intrinsic_rewards = []
    episodic_intrinsic_rewards_max = []
    episodic_returns_this_update = []
    successes_this_update = 0
    failures_this_update = 0
    timeouts_this_update = 0

    for t in range(s.cfg.num_steps):
        s.global_step += s.num_envs

        with torch.no_grad():
            action, logprob, entropy, value = s.model.act(s.obs_t)
            pred_next_obs, pred_reward = s.world_model(s.obs_t, action)

        next_obs, reward, terminated, truncated, infos = s.envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)

        # Per-step head-bank trigger (LOOP-0014): model-free success-collapse detector
        # on this step's episode terminations.
        _head_bank_step_check(s, reward, done)

        # Increment shared regime step counter by the number of parallel env steps taken
        if hasattr(s, 'regime_step_counter'):
            s.regime_step_counter[0] += s.num_envs

        # Log regime_id for regime switch visualization
        current_env_regime = s.current_mode_regime
        if "regime_id" in infos:
            regime_ids = infos["regime_id"]
            # Filter out None values that might appear due to env resets
            valid_regimes = [r for r in regime_ids if r is not None]
            if valid_regimes:
                current_env_regime = int(np.bincount(np.array(valid_regimes, dtype=int)).argmax())  # majority vote
                s.logger.scalar("charts/regime_id", float(np.mean(valid_regimes)), s.global_step)

        # Handle regime switch
        if current_env_regime != s.current_mode_regime:
            print(f"[{s.global_step}] Regime switch detected! {s.current_mode_regime} -> {current_env_regime}. Snapshotting anchor model.")
            prev_regime = s.current_mode_regime
            s.current_mode_regime = current_env_regime
            # O2 policy-swap topline (research note 0005): bank the outgoing regime's
            # learner; restore the incoming regime's if we've seen it before. The restore
            # lands mid-rollout, so this update's PPO phase mixes two policies — an
            # accepted, conservative imprecision for a ceiling estimate. Restore precedes
            # the anchor snapshot so the anchor tracks the restored policy.
            if s.policy_swap_topline:
                s.policy_swap_snapshots[prev_regime] = _snapshot_learner(s, s.policy_swap_scope)
                snap = s.policy_swap_snapshots.get(current_env_regime)
                if snap is not None:
                    _restore_learner(s, snap)
                s.logger.scalar(
                    "brain_neuromod/policy_swap_restored",
                    float(snap is not None),
                    s.global_step,
                )
            # O1 oracle-timed critic damp (research note 0005): arm the ground-truth-timed
            # damp window and resync now so the damp already covers this update's PPO phases.
            if s.cfg.critic_lr_oracle_scale > 0.0:
                s.cfg.critic_oracle_remaining = CRITIC_ORACLE_UPDATES
                apply_inner_lr(s.optimizer, s.optimizer.param_groups[0]["lr"], s.cfg)
            # G3 head bank, oracle trigger (LOOP-0012): bank the outgoing regime's heads,
            # select the incoming regime's slot by ground-truth id.
            if s.head_bank_slots > 0 and s.head_bank_trigger == "oracle":
                _head_bank_switch(s, current_env_regime % s.head_bank_slots)
            if s.anchor_model is not None:
                s.anchor_model.load_state_dict(s.model.state_dict())

        # Handle autoreset: use final_observation for surprise calc on done envs
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            final_obs_mask = infos.get("_final_observation", done)
            for i, is_final in enumerate(final_obs_mask):
                if is_final and i < len(infos["final_observation"]):
                     real_next_obs[i] = infos["final_observation"][i]

        real_next_obs_t = torch.tensor(real_next_obs, dtype=torch.float32, device=s.device)

        # Compute intrinsic reward (surprise signal)
        with torch.no_grad():
            target_indices = torch.argmax(real_next_obs_t, dim=1)
            state_loss = torch.nn.functional.cross_entropy(
                pred_next_obs, target_indices, reduction='none'
            )
            state_surprise = state_loss.mean(dim=[1, 2])

            s.logger.scalar("debug/raw_cross_entropy_loss", state_surprise.mean().item(), s.global_step)

            real_reward_t = torch.tensor(reward, dtype=torch.float32, device=s.device)
            reward_surprise = torch.nn.functional.mse_loss(
                pred_reward, real_reward_t, reduction='none'
            )

            if s.cfg.mode == "passive":
                total_surprise = torch.zeros_like(state_surprise)
            else:
                total_surprise = state_surprise + reward_surprise

            raw_intrinsic = total_surprise * s.intrinsic_coef
            intrinsic_reward = torch.clamp(raw_intrinsic, 0.0, s.intrinsic_reward_clip)

            s.logger.scalar("debug/wm_raw_error_mean", total_surprise.mean().item(), s.global_step)
            s.logger.scalar("debug/intrinsic_reward_raw", raw_intrinsic.mean().item(), s.global_step)

            episodic_intrinsic_rewards.append(intrinsic_reward.mean().item())
            episodic_intrinsic_rewards_max.append(intrinsic_reward.max().item())

        # Update per-env episode trackers
        s.running_returns += reward
        s.running_lengths += 1

        # Store transition (extrinsic + intrinsic reward)
        extrinsic_reward = torch.tensor(reward, dtype=torch.float32, device=s.device)
        total_reward = extrinsic_reward + intrinsic_reward

        s.buffer.add(
            obs=s.obs_t,
            actions=action,
            logprobs=logprob,
            rewards=total_reward,
            dones=torch.tensor(done, dtype=torch.float32, device=s.device),
            values=value,
            next_obs=real_next_obs_t,
            extrinsic_rewards=extrinsic_reward,
        )

        s.obs_t = torch.tensor(next_obs, dtype=torch.float32, device=s.device)

        # Log finished episodes
        if np.any(done):
            done_indices = np.where(done)[0]
            for i in done_indices:
                s.logger.scalar("charts/episodic_return", s.running_returns[i], s.global_step)
                s.logger.scalar("charts/episodic_length", s.running_lengths[i], s.global_step)
                episodic_returns_this_update.append(s.running_returns[i])

                # Extract Goal Outcomes
                outcome = 0  # timeout/other
                reached_good = infos["reached_good_goal"][i] if "reached_good_goal" in infos else 0.0
                reached_bad = infos["reached_bad_goal"][i] if "reached_bad_goal" in infos else 0.0

                if reached_good > 0:
                    outcome = 1  # success
                elif reached_bad > 0:
                    outcome = -1  # failure

                s.outcome_window.append(outcome)
                s.recent_outcome_window.append(outcome)

                if outcome == 1:
                    successes_this_update += 1
                elif outcome == -1:
                    failures_this_update += 1
                else:
                    timeouts_this_update += 1

                if len(s.outcome_window) > 0:
                    success_rate = sum(1 for x in s.outcome_window if x == 1) / len(s.outcome_window)
                    failure_rate = sum(1 for x in s.outcome_window if x == -1) / len(s.outcome_window)
                    timeout_rate = sum(1 for x in s.outcome_window if x == 0) / len(s.outcome_window)

                    s.logger.scalar("charts/success_rate", success_rate, s.global_step)
                    s.logger.scalar("charts/failure_rate", failure_rate, s.global_step)
                    s.logger.scalar("charts/timeout_rate", timeout_rate, s.global_step)

                if "regime_id" in infos:
                    regime = infos["regime_id"][i]
                    s.logger.scalar(f"charts/r_regime_{regime}", s.running_returns[i], s.global_step)

                s.running_returns[i] = 0
                s.running_lengths[i] = 0

    # GAE computation
    with torch.no_grad():
        _, last_value = s.model.forward(s.obs_t)
    s.buffer.compute_returns_and_advantages(last_value, s.cfg.gamma, s.cfg.gae_lambda)

    # =====================================================================
    # Phase B: Update policy on real data
    # =====================================================================

    update_stats = []
    for epoch in range(s.cfg.update_epochs):
        minibatches = s.buffer.get_minibatches(s.cfg.minibatch_size, shuffle=True)
        for obs, actions, logprobs, advantages, returns, values, _, _ in minibatches:
            ppo_batch = [obs, actions, logprobs, advantages, returns, values]
            
            anchor_logprobs_list = None
            if s.cfg.anchoring_weight > 0.0 and s.anchor_model is not None:
                with torch.no_grad():
                    anchor_logprobs, _, _ = s.anchor_model.evaluate_actions(obs, actions)
                anchor_logprobs_list = [anchor_logprobs]

            stats = ppo_update(s.model, s.optimizer, [ppo_batch], s.cfg, anchor_logprobs_list)
            update_stats.append(stats)

    # Phase B2: Replay from episodic memory (Brain-controlled)
    replay_stats = []
    if s.replay_ratio > 0.01 and s.episodic_memory is not None and s.episodic_memory.size >= s.cfg.minibatch_size:
        n_replay = max(1, int(s.cfg.minibatch_size * s.replay_ratio))
        n_fresh = s.cfg.minibatch_size - n_replay

        replay_sample = s.episodic_memory.sample(n_replay, prioritization=s.replay_prioritization, current_regime=s.current_mode_regime)
        if replay_sample is not None:
            # Get fresh samples from current buffer
            fresh_idxs = np.random.randint(0, s.cfg.num_steps * s.num_envs, size=n_fresh)
            flat_obs = s.buffer.obs.reshape((-1,) + s.obs_shape)
            flat_actions = s.buffer.actions.reshape(-1)
            flat_rewards = s.buffer.extrinsic_rewards.reshape(-1)
            flat_next_obs = s.buffer.next_obs.reshape((-1,) + s.obs_shape)

            # Combine fresh + replay observations and actions
            mixed_obs = torch.cat([flat_obs[fresh_idxs], replay_sample["obs"]], dim=0)
            mixed_actions = torch.cat([flat_actions[fresh_idxs], replay_sample["actions"]], dim=0)

            # Recompute values & logprobs under current policy for the mixed batch
            with torch.no_grad():
                _, mixed_values = s.model.forward(mixed_obs)

            # Compute simple 1-step returns: r + gamma * V(s') for replay
            mixed_rewards = torch.cat([flat_rewards[fresh_idxs], replay_sample["rewards"]], dim=0)
            mixed_next_obs = torch.cat([flat_next_obs[fresh_idxs], replay_sample["next_obs"]], dim=0)
            with torch.no_grad():
                _, next_values = s.model.forward(mixed_next_obs)
            mixed_dones = torch.cat([
                s.buffer.dones.reshape(-1)[fresh_idxs],
                replay_sample["dones"]
            ], dim=0)
            mixed_returns = mixed_rewards + s.cfg.gamma * next_values * (1.0 - mixed_dones)
            mixed_advantages = mixed_returns - mixed_values
            mixed_advantages = (mixed_advantages - mixed_advantages.mean()) / (mixed_advantages.std(unbiased=False) + 1e-8)

            # Compute old logprobs under current policy BEFORE the update
            # (detached so they remain fixed as the "old" reference)
            with torch.no_grad():
                old_logprobs, _, _ = s.model.evaluate_actions(mixed_obs, mixed_actions)

            ppo_batch = [mixed_obs, mixed_actions, old_logprobs, mixed_advantages, mixed_returns, mixed_values]
            
            anchor_logprobs_list = None
            if s.cfg.anchoring_weight > 0.0 and s.anchor_model is not None:
                with torch.no_grad():
                    anchor_logprobs, _, _ = s.anchor_model.evaluate_actions(mixed_obs, mixed_actions)
                anchor_logprobs_list = [anchor_logprobs]

            rs = ppo_update(s.model, s.optimizer, [ppo_batch], s.cfg, anchor_logprobs_list)
            replay_stats.append(rs)

    # =====================================================================
    # Phase C: Train World Model
    # =====================================================================

    wm_stats = []
    for epoch in range(s.cfg.update_epochs):
        minibatches = s.buffer.get_minibatches(s.cfg.minibatch_size, shuffle=True)
        for obs, actions, _, _, _, _, next_obs, rewards in minibatches:
            pred_next_obs, pred_reward = s.world_model(obs, actions)
            target_indices = torch.argmax(next_obs, dim=1)
            loss_state = torch.nn.functional.cross_entropy(pred_next_obs, target_indices)
            loss_reward = torch.nn.functional.mse_loss(pred_reward, rewards)
            wm_loss = loss_state + loss_reward

            s.wm_optimizer.zero_grad()
            wm_loss.backward()
            s.wm_optimizer.step()

            wm_stats.append({
                "world_model/loss_total": wm_loss.item(),
                "world_model/loss_state": loss_state.item(),
                "world_model/loss_reward": loss_reward.item(),
            })

    # Archive current rollout to episodic memory (using extrinsic rewards only
    # to avoid stale intrinsic bonuses biasing replay)
    if s.episodic_memory is not None:
        s.episodic_memory.store_from_rollout(
            s.buffer, regime_ids=s.current_mode_regime,
            use_extrinsic_rewards=True,
        )

    # =====================================================================
    # Phase D: Dream & update policy on imagined data
    # =====================================================================

    dream_stats = []
    dream_buffer = None
    if s.imagined_horizon > 0:
        rand_time_idxs = torch.randint(0, s.cfg.num_steps, (s.num_envs,), device=s.device)
        env_idxs = torch.arange(s.num_envs, device=s.device)
        start_states = s.buffer.obs[rand_time_idxs, env_idxs]

        imagined_trajectories = s.world_model.generate_imagined_trajectories(
            policy_net=s.model,
            start_states=start_states,
            horizon=s.imagined_horizon
        )

        dream_buffer = RolloutBuffer(s.imagined_horizon, s.num_envs, s.obs_shape, s.device)
        for t_step, traj in enumerate(imagined_trajectories):
            dream_buffer.add(
                obs=traj["obs"],
                actions=traj["actions"],
                logprobs=traj["logprobs"],
                rewards=traj["rewards"],
                dones=traj["dones"],
                values=traj["values"],
                next_obs=traj["next_obs"]
            )

        if imagined_trajectories:
            last_dream_obs = imagined_trajectories[-1]["next_obs"]
            with torch.no_grad():
                _, last_dream_value = s.model.forward(last_dream_obs)
            dream_buffer.compute_returns_and_advantages(last_dream_value, s.cfg.gamma, s.cfg.gae_lambda)

        for epoch in range(1):
            minibatches = dream_buffer.get_minibatches(s.cfg.minibatch_size, shuffle=True)
            for obs, actions, logprobs, advantages, returns, values, _, _ in minibatches:
                ppo_batch = [obs, actions, logprobs, advantages, returns, values]
                
                anchor_logprobs_list = None
                if s.cfg.anchoring_weight > 0.0 and s.anchor_model is not None:
                    with torch.no_grad():
                        anchor_logprobs, _, _ = s.anchor_model.evaluate_actions(obs, actions)
                    anchor_logprobs_list = [anchor_logprobs]

                ds = ppo_update(s.model, s.optimizer, [ppo_batch], s.cfg, anchor_logprobs_list)
                dream_stats.append(ds)

    # =====================================================================
    # Logging
    # =====================================================================

    avg_stats = {k: np.mean([st[k] for st in update_stats]) for k in update_stats[0]} if update_stats else {}
    avg_wm_stats = {k: np.mean([st[k] for st in wm_stats]) for k in wm_stats[0]} if wm_stats else {}

    # LOOP-0007 cand 4: restore the Brain's base coefs (undo this update's transient spike),
    # then run the change-point detector on this update's TD-error surprise (value_loss) to
    # arm the spike for upcoming updates if a regime switch is inferred.
    s.cfg.ent_coef = base_ent_coef
    s.intrinsic_coef = base_intrinsic_coef
    if s.surprise_spike_threshold > 0.0:
        _update_surprise_spike(s, float(avg_stats.get("loss/value", 0.0)))
        s.logger.scalar("brain_neuromod/surprise_spike_active", float(spike_active), s.global_step)
    # G3 head bank, surprise trigger (A-R1, LOOP-0012): the learned WHEN — an independent
    # value-loss change-point detector drives bank/restore instead of the ground-truth switch.
    if s.head_bank_slots > 0 and s.head_bank_trigger == "surprise":
        _head_bank_surprise_check(s, float(avg_stats.get("loss/value", 0.0)))

    # LOOP-0007 cand 2: periodic ReDo reset of dormant head units; log the dormant-fraction probe.
    if s.redo_interval > 0 and update % s.redo_interval == 0:
        redo_fractions = redo_reset_heads(s.model, s.optimizer, s.obs_t)
        for head, frac in redo_fractions.items():
            s.logger.scalar(f"brain_neuromod/redo_dormant_fraction_{head}", frac, s.global_step)

    for k, v in avg_stats.items():
        s.logger.scalar(k, v, s.global_step)
    for k, v in avg_wm_stats.items():
        s.logger.scalar(k, v, s.global_step)

    if dream_stats and dream_buffer is not None:
        avg_dream_stats = {f"ppo/imagined_{k.split('/')[-1]}": np.mean([ds[k] for ds in dream_stats]) for k in dream_stats[0]}
        for k, v in avg_dream_stats.items():
            s.logger.scalar(k, v, s.global_step)
        s.logger.scalar("ppo/imagined_value_mean", dream_buffer.values.mean().item(), s.global_step)
        s.logger.scalar("ppo/imagined_return_mean", dream_buffer.returns.mean().item(), s.global_step)

    mean_intrinsic = np.mean(episodic_intrinsic_rewards) if episodic_intrinsic_rewards else 0.0
    max_intrinsic = np.max(episodic_intrinsic_rewards_max) if episodic_intrinsic_rewards_max else 0.0
    buffer_rewards_abs_mean = s.buffer.rewards.abs().mean().item()

    s.logger.scalar("ppo/intrinsic_reward_mean", mean_intrinsic, s.global_step)
    s.logger.scalar("ppo/intrinsic_reward_max", max_intrinsic, s.global_step)

    if buffer_rewards_abs_mean > 1e-6:
        s.logger.scalar("ppo/intrinsic_reward_ratio", mean_intrinsic / buffer_rewards_abs_mean, s.global_step)
    else:
        s.logger.scalar("ppo/intrinsic_reward_ratio", 0.0, s.global_step)

    s.logger.scalar("charts/learning_rate", lrnow, s.global_step)
    s.logger.scalar("charts/heartbeat", s.global_step, s.global_step)
    s.logger.scalar("charts/reward_step_mean", s.buffer.rewards.mean().item(), s.global_step)
    s.logger.scalar("charts/reward_step_max", s.buffer.rewards.max().item(), s.global_step)
    s.logger.scalar("charts/reward_step_std", s.buffer.rewards.std(unbiased=False).item(), s.global_step)
    s.logger.scalar("brain/ent_coef", s.cfg.ent_coef, s.global_step)
    s.logger.scalar("brain/intrinsic_coef", s.intrinsic_coef, s.global_step)
    s.logger.scalar("brain/imagined_horizon", float(s.imagined_horizon), s.global_step)
    s.logger.scalar("brain/anchoring_weight", float(s.cfg.anchoring_weight), s.global_step)

    sps = int(s.global_step / max(1e-9, (time.time() - s.start_time)))
    s.logger.scalar("charts/SPS", sps, s.global_step)
    s.logger.scalar("charts/replay_ratio", s.replay_ratio, s.global_step)
    s.logger.scalar("charts/replay_prioritization", s.replay_prioritization, s.global_step)
    s.logger.scalar("charts/episodic_memory_fullness", s.episodic_memory.fullness if s.episodic_memory else 0.0, s.global_step)

    # Save periodic checkpoints, and a final one only when periodic saving is actually enabled.
    # meta_env signals "no inner checkpoints" via save_every_updates=9999 (> num_updates); without
    # this guard the `update == num_updates` case still force-saved a 54 MB checkpoint every episode
    # (~12 GB per sweep) despite checkpoints being off. Enable with save_checkpoints=True.
    save_final = update == s.num_updates and s.save_every_updates <= s.num_updates
    if update % s.save_every_updates == 0 or save_final:
        ckpt_path = os.path.join(s.save_dir, f"{s.run_name}_update{update}.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": s.model.state_dict(),
                "optimizer_state_dict": s.optimizer.state_dict(),
                "world_model_state_dict": s.world_model.state_dict(),
                "wm_optimizer_state_dict": s.wm_optimizer.state_dict(),
                "cfg": s.cfg.__dict__,
                "global_step": s.global_step,
            },
            ckpt_path,
        )
        print(f"[save] {ckpt_path}")

    # Log progress every 10 updates or at the very end
    if update % 10 == 0 or update == s.num_updates:
        print(f"update {update}/{s.num_updates} | step={s.global_step} | SPS={sps}")
        
    s.current_update += 1
    
    # Plot only at the very end of the run
    if s.current_update > s.num_updates and s.logger is not None:
        plot_dir = getattr(s.logger, 'full_dir', "")
        if not plot_dir:
            # Fallback if full_dir not set
            plot_dir = s.save_dir.replace("inner_checkpoints", "")
        # Remove trailing slash if any and ensure it exists
        plot_dir = plot_dir.rstrip("\\/")
        os.makedirs(plot_dir, exist_ok=True)
        s.logger.plot(save_dir=plot_dir, title=f"Inner Agent: {s.run_name}")

    # -----------------------------------------------------------------
    # Build stats dict for the Brain
    # -----------------------------------------------------------------

    # Outcome rates from the rolling windows
    n_outcomes = len(s.outcome_window)
    success_rate = sum(1 for x in s.outcome_window if x == 1) / max(n_outcomes, 1)
    failure_rate = sum(1 for x in s.outcome_window if x == -1) / max(n_outcomes, 1)

    n_recent_outcomes = len(s.recent_outcome_window)
    recent_success_rate = sum(1 for x in s.recent_outcome_window if x == 1) / max(n_recent_outcomes, 1)
    recent_failure_rate = sum(1 for x in s.recent_outcome_window if x == -1) / max(n_recent_outcomes, 1)
    recent_timeout_rate = sum(1 for x in s.recent_outcome_window if x == 0) / max(n_recent_outcomes, 1)

    mean_return = float(np.mean(episodic_returns_this_update)) if episodic_returns_this_update else 0.0

    return {
        "done": update >= s.num_updates,
        "global_step": s.global_step,
        "update": update,

        # Performance signals
        "mean_episodic_return": mean_return,
        "episodic_return_sum": float(np.sum(episodic_returns_this_update)) if episodic_returns_this_update else 0.0,
        "completed_episodes": len(episodic_returns_this_update),
        "successful_episodes": successes_this_update,
        "failed_episodes": failures_this_update,
        "timeout_episodes": timeouts_this_update,
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "recent_success_rate": recent_success_rate,
        "recent_failure_rate": recent_failure_rate,
        "recent_timeout_rate": recent_timeout_rate,

        # Surprise / world model signals
        "mean_surprise": float(np.mean(episodic_intrinsic_rewards)) if episodic_intrinsic_rewards else 0.0,
        "wm_loss_state": avg_wm_stats.get("world_model/loss_state", 0.0),
        "wm_loss_reward": avg_wm_stats.get("world_model/loss_reward", 0.0),

        # Policy signals
        "policy_entropy": avg_stats.get("loss/entropy", 0.0),
        "policy_loss": avg_stats.get("loss/policy", 0.0),
        "value_loss": avg_stats.get("loss/value", 0.0),

        # Current hyperparameter values
        "current_lr": lrnow,
        "current_ent_coef": s.cfg.ent_coef,
        "current_intrinsic_coef": s.intrinsic_coef,
        "current_replay_ratio": s.replay_ratio,
        "current_replay_prioritization": s.replay_prioritization,
        "current_anchoring_weight": s.cfg.anchoring_weight,
        "episodic_memory_size": s.episodic_memory.size if s.episodic_memory else 0,
        "episodic_memory_fullness": s.episodic_memory.fullness if s.episodic_memory else 0.0,
        "current_imagined_horizon": s.imagined_horizon,
        "current_mode_regime": s.current_mode_regime,
    }


def close_inner_training(state: InnerTrainState):
    """Clean up environments and logger."""
    state.envs.close()
    if state.logger is not None:
        state.logger.close()


# =========================================================================
# Original train_ppo() — now delegates to init/step functions
# =========================================================================

def train_ppo(
    env_id: str,
    cfg: PPOConfig,
    *,
    steps_per_regime: int | None = None,
    episodes_per_regime: int | None = None,
    start_regime: int = 0,
    num_regimes: int = 2,
    run_name: str | None = None,
    save_dir: str = "checkpoints",
    save_every_updates: int = 50,
    anneal_lr: bool = True,
    resume_path: str | None = None,
    intrinsic_coef: float = 0.1,
    intrinsic_reward_clip: float = 0.1,
    imagined_horizon: int = 5,
    wm_lr: float = 1e-4,
    replay_ratio: float = 0.0,
    replay_prioritization: float = 0.0,
):
    """
    Main Dyna-PPO training loop.

    Each update cycle has three phases:
        A) Collect real experience (with intrinsic curiosity reward)
        B) Train World Model on real transitions (supervised)
        C) Generate imagined trajectories and update policy on dreams
    """

    state = init_inner_training(
        env_id=env_id,
        cfg=cfg,
        steps_per_regime=steps_per_regime,
        episodes_per_regime=episodes_per_regime,
        start_regime=start_regime,
        num_regimes=num_regimes,
        run_name=run_name,
        save_dir=save_dir,
        save_every_updates=save_every_updates,
        anneal_lr=anneal_lr,
        resume_path=resume_path,
        intrinsic_coef=intrinsic_coef,
        intrinsic_reward_clip=intrinsic_reward_clip,
        imagined_horizon=imagined_horizon,
        wm_lr=wm_lr,
        replay_ratio=replay_ratio,
        replay_prioritization=replay_prioritization,
    )

    print(f"Training on {state.device} with {state.num_envs} envs "
          f"for {state.num_updates} updates (starting from update {state.current_update}).")

    while True:
        stats = run_inner_update(state)
        if stats["done"]:
            break

    close_inner_training(state)
