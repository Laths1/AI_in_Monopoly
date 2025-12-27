import csv
import os
import math
import numpy as np
from collections import defaultdict


class MonopolyCSVLogger:
    """
    Per-environment CSV logger for Monopoly RL.
    Safe for SubprocVecEnv (one file per process).
    """

    def __init__(self, log_dir="logs", env_id=None):
        os.makedirs(log_dir, exist_ok=True)

        pid = os.getpid()
        suffix = f"_env{env_id}" if env_id is not None else ""
        self.path = os.path.join(log_dir, f"monopoly_stats_pid{pid}{suffix}.csv")

        self.episode_id = 0
        self.turn = 0
        self._initialized = False

    # ----------------------------
    # Initialization
    # ----------------------------
    def _init_file(self):
        if self._initialized:
            return

        write_header = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "episode", "turn",
                    "player_id", "is_agent",
                    "action", "reward",
                    "invalid_action",
                    "action_mask_entropy",
                    "cash", "net_worth",
                    "num_assets",
                    "properties", "railways", "utilities",
                    "houses", "hotels",
                    "mortgaged_assets",
                    "position",
                    "in_jail", "jail_turns",
                    "full_sets",
                    "trades_made",
                    "auctions_won",
                    "winner",
                    "agent_won",
                    "episode_length"
                ])
        self._initialized = True

    # ----------------------------
    # Episode lifecycle
    # ----------------------------
    def start_episode(self):
        self._init_file()
        self.episode_id += 1
        self.turn = 0

    def step(self):
        self.turn += 1

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def action_mask_entropy(mask):
        probs = mask / (mask.sum() + 1e-8)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return float(entropy)

    @staticmethod
    def player_stats(player):
        props = rails = utils = houses = hotels = mort = full_sets = 0

        for a in player.assets:
            if hasattr(a, "property_set"):
                props += 1
                houses += getattr(a, "houses", 0)
                hotels += int(getattr(a, "hotel", False))
                if a.has_full_set:
                    full_sets += 1
            if a.__class__.__name__ == "Railway":
                rails += 1
            if a.__class__.__name__ == "Utility":
                utils += 1
            if getattr(a, "is_mortgaged", False):
                mort += 1

        return {
            "cash": player.capital,
            "net_worth": player.capital + getattr(player, "net_asset_value", 0),
            "num_assets": len(player.assets),
            "properties": props,
            "railways": rails,
            "utilities": utils,
            "houses": houses,
            "hotels": hotels,
            "mortgaged": mort,
            "position": player.position,
            "in_jail": int(player.in_jail),
            "jail_turns": getattr(player, "jail_counter", 0),
            "full_sets": full_sets,
            "trades": getattr(player, "trades_made", 0),
            "auctions": getattr(player, "auctions_won", 0)
        }

    # ----------------------------
    # Logging
    # ----------------------------
    def log_step(
        self,
        players,
        agent_index,
        action,
        reward,
        action_mask,
        invalid_action,
        done=False,
        winner_index=None
    ):
        entropy = self.action_mask_entropy(action_mask)

        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)

            for i, p in enumerate(players):
                stats = self.player_stats(p)
                writer.writerow([
                    self.episode_id,
                    self.turn,
                    p.player_name,
                    int(i == agent_index),
                    action if i == agent_index else "heuristic",
                    reward if i == agent_index else 0.0,
                    int(invalid_action and i == agent_index),
                    entropy,
                    stats["cash"],
                    stats["net_worth"],
                    stats["num_assets"],
                    stats["properties"],
                    stats["railways"],
                    stats["utilities"],
                    stats["houses"],
                    stats["hotels"],
                    stats["mortgaged"],
                    stats["position"],
                    stats["in_jail"],
                    stats["jail_turns"],
                    stats["full_sets"],
                    stats["trades"],
                    stats["auctions"],
                    winner_index if done else "",
                    int(done and winner_index == agent_index),
                    self.turn if done else ""
                ])
