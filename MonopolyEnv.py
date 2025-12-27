# MonopolyEnv.py
"""
MonopolyEnv - Gymnasium-compatible environment for Monopoly with proper action masking.

Key improvements:
- Action masking to prevent invalid actions
- Better reward shaping with net worth tracking
- Cleaner separation of agent and heuristic player logic
- Proper handling of all 16 actions
- Episode termination conditions
"""

import random
import time
import threading
from typing import Tuple, Optional, Dict, Any, List
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from monopoly_board import *
from bank import *
from trading_floor import *
from player import *   
from auction_floor import * 
from cards import *
from rewards import *
from actions import *
from turn_timer import Timer90
from log_messages import *
import pandas as pd
from monopoly_to_csv import MonopolyCSVLogger

# Toggles to see actions printed
verbose = False

def log(message):
    """Global logging function."""
    if verbose:
        print(message)

# --------------------------
# Observation helpers
# --------------------------
def flatten_board_obs(board_info: dict, board: Board) -> np.ndarray:
    """Convert board state to fixed numeric vector."""
    prop_names = []
    rail_names = []
    util_names = []
    for pos in range(40):
        sq = board.board[pos]
        if sq.type == "Property" and sq.obj:
            prop_names.append(sq.obj.property_name)
        elif sq.type == "Railway" and sq.obj:
            rail_names.append(sq.obj.property_name)
        elif sq.type == "Utility" and sq.obj:
            util_names.append(sq.obj.property_name)

    def indicators(names, unsold_key, sold_key, mort_key, price_key):
        unsold = np.array([1.0 if n in board_info.get(unsold_key, []) else 0.0 for n in names], dtype=np.float32)
        sold = np.array([1.0 if n in board_info.get(sold_key, []) else 0.0 for n in names], dtype=np.float32)
        mort = np.array([1.0 if n in board_info.get(mort_key, []) else 0.0 for n in names], dtype=np.float32)
        prices = np.array([board_info.get(price_key, {}).get(n, 0.0) for n in names], dtype=np.float32)
        return unsold, sold, mort, prices

    unsold_p, sold_p, mort_p, prices_p = indicators(
        prop_names, "unsold_properties", "sold_properties", "mortgaged_properties", "property_prices"
    )
    unsold_r, sold_r, mort_r, prices_r = indicators(
        rail_names, "unsold_railways", "sold_railways", "mortgaged_railways", "railway_prices"
    )
    unsold_u, sold_u, mort_u, prices_u = indicators(
        util_names, "unsold_utilities", "sold_utilities", "mortgaged_utilities", "utility_prices"
    )

    # Normalize prices
    max_price = 1.0
    candidates = []
    if prices_p.size:
        candidates.append(prices_p.max())
    if prices_r.size:
        candidates.append(prices_r.max())
    if prices_u.size:
        candidates.append(prices_u.max())
    if candidates:
        max_price = float(max(candidates))
        if max_price <= 0:
            max_price = 1.0

    prices_p = prices_p / max_price if prices_p.size else prices_p
    prices_r = prices_r / max_price if prices_r.size else prices_r
    prices_u = prices_u / max_price if prices_u.size else prices_u

    vec = np.concatenate([
        unsold_p, sold_p, mort_p, prices_p,
        unsold_r, sold_r, mort_r, prices_r,
        unsold_u, sold_u, mort_u, prices_u
    ]).astype(np.float32)

    return vec


def flatten_player_obs(board: Board, player: Player) -> np.ndarray:
    """
    Build numeric vector for player state.
    Layout: [cash_norm, pos_norm, in_jail, jail_free] + per-tile info (40 tiles × 4 features)
    """
    cash_norm = float(player.capital) / 5000.0
    pos_norm = float(player.position) / 39.0
    in_jail = 1.0 if player.in_jail else 0.0
    jail_free = 1.0 if getattr(player, "has_jail_free_card", False) else 0.0

    player_vec = [cash_norm, pos_norm, in_jail, jail_free]

    asset_vec = []
    for pos in range(40):
        sq = board.board[pos]
        ownership = 0.0
        mortgaged = 0.0
        houses_norm = 0.0
        hotel_flag = 0.0

        if sq.type in ["Property", "Railway", "Utility"] and sq.obj:
            asset = sq.obj
            if not asset.is_sold():
                ownership = 0.0
            else:
                ownership = 1.0 if asset.owner == player else 2.0

            mortgaged = 1.0 if getattr(asset, "is_mortgaged", False) else 0.0
            houses_norm = float(getattr(asset, "houses", 0)) / 4.0
            hotel_flag = 1.0 if getattr(asset, "hotel", False) else 0.0

        asset_vec.extend([ownership, mortgaged, houses_norm, hotel_flag])

    obs = np.array(player_vec + asset_vec, dtype=np.float32)
    return obs


def flatten_turn_info(turn_info: Optional[Dict[str, Any]]) -> np.ndarray:
    """Simple 4D turn vector: dice_roll, old_pos, new_pos, capital (all normalized)."""
    if turn_info is None:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    dice_roll = float(turn_info.get("dice_roll", 0)) / 6.0
    old_position = float(turn_info.get("old_position", 0)) / 39.0
    new_position = float(turn_info.get("new_position", 0)) / 39.0
    capital = float(turn_info.get("capital", 0)) / 5000.0

    return np.array([dice_roll, old_position, new_position, capital], dtype=np.float32)


# --------------------------
# Game mechanics helpers
# --------------------------
def roll_dice():
    return random.randint(1, 6)


def resolve_card(player, card, board, bank, players, dice_value):
    """Execute card effects."""
    match card.action:
        case "move":
            player.position = card.amount
            log(MESSAGES.MOVE_TO_POSITION.format(player_name=player.player_name, amount=card.amount))
        case "move_back":
            player.position = (player.position - card.amount) % 40
            log(MESSAGES.MOVE_BACK.format(player_name=player.player_name, amount=card.amount, position=player.position))
        case "jail":
            player.position = 10
            if player.has_jail_free_card:
                player.has_jail_free_card = False
                log(MESSAGES.USES_JAIL_CARD.format(player_name=player.player_name))
            else:
                player.in_jail = True
                log(MESSAGES.GOES_TO_JAIL.format(player_name=player.player_name))
        case "pay_bank":
            player.pay(card.amount)
            bank.buy(card.amount)
            log(MESSAGES.PAYS_TO_BANK.format(player_name=player.player_name, amount=card.amount))
        case "gain_bank":
            player.receive(card.amount)
            bank.sell(card.amount)
            log(MESSAGES.RECEIVES_FROM_BANK.format(player_name=player.player_name, amount=card.amount))
        case "jail_free":
            player.has_jail_free_card = True
            log(MESSAGES.RECEIVES_JAIL_CARD.format(player_name=player.player_name))
        case "repair":
            house_cost = card.target["house"]
            hotel_cost = card.target["hotel"]
            total = 0
            for a in player.show_assets():
                if isinstance(a, Property):
                    total += a.houses * house_cost
                    total += a.hotel * hotel_cost
                    log(MESSAGES.REPAIRS_PROPERTIES.format(player_name=player.player_name, total=total))
            player.pay(total)
            bank.buy(total)
        case "nearest_railway":
            rail_positions = [5, 15, 25, 35]
            new_pos = min(rail_positions, key=lambda p: (p - player.position) % 40)
            player.position = new_pos
            log(MESSAGES.MOVES_TO_NEAREST.format(player_name=player.player_name, tile_type="Railway", new_pos=new_pos))
        case "nearest_utility":
            util_positions = [12, 28]
            new_pos = min(util_positions, key=lambda p: (p - player.position) % 40)
            player.position = new_pos
            log(MESSAGES.MOVES_TO_NEAREST.format(player_name=player.player_name, tile_type="Utility", new_pos=new_pos))

def tile_interaction(current_player, tile, bank, board, players, dice, chance_deck, community_deck):
    """Auto-handle tile landing (for movement phase)."""
    if tile.get_type() == "Property":
        prop = tile.get_obj()
        if prop.sold and prop.owner != current_player:
            rent = prop.pay_rent()
            current_player.pay(rent)
            prop.owner.receive(rent)
            log(MESSAGES.PAYS_RENT.format(player_name=current_player.player_name, rent=rent, owner_name=prop.owner.player_name))

    elif tile.get_type() == "Railway":
        rail = tile.get_obj()
        if rail.sold and rail.owner != current_player:
            rent = rail.pay_rent()
            current_player.pay(rent)
            rail.owner.receive(rent)
            log(MESSAGES.PAYS_RENT.format(player_name=current_player.player_name, rent=rent, owner_name=rail.owner.player_name))

    elif tile.get_type() == "Utility":
        util = tile.get_obj()
        if util.sold and util.owner != current_player:
            rent = util.pay_rent(dice)
            current_player.pay(rent)
            util.owner.receive(rent)
            log(MESSAGES.PAYS_RENT.format(player_name=current_player.player_name, rent=rent, owner_name=util.owner.player_name))

    elif tile.get_type() == "Special":
        if tile.obj == "Income Tax":
            current_player.pay(200)
            bank.buy(200)
            log(MESSAGES.PAYS_TAX.format(player_name=current_player.player_name, amount=200, tax_type="Income"))
        elif tile.obj == "Luxury Tax":
            current_player.pay(100)
            bank.buy(100)
            log(MESSAGES.PAYS_TAX.format(player_name=current_player.player_name, amount=100, tax_type="Luxury"))
        elif tile.obj == "Go To Jail":
            current_player.position = 10
            if current_player.has_jail_free_card:
                current_player.has_jail_free_card = False
                log(MESSAGES.USES_JAIL_CARD.format(player_name=current_player.player_name))
            else:
                current_player.in_jail = True
                log(MESSAGES.GOES_TO_JAIL.format(player_name=current_player.player_name))
        elif tile.obj == "Chance":
            card = chance_deck.draw()
            log(MESSAGES.DRAWS_CHANCE.format(player_name=current_player.player_name, description=card.description))
            resolve_card(current_player, card, board, bank, players, dice)
        elif tile.obj == "Community Chest":
            card = community_deck.draw()
            log(MESSAGES.DRAWS_COMMUNITY.format(player_name=current_player.player_name, description=card.description))
            resolve_card(current_player, card, board, bank, players, dice)


def check_set_ownership(player):
    """Check full color sets and mark properties."""
    player_properties = [asset for asset in player.show_assets() if isinstance(asset, Property)]

    counts = {
        "Brown": 0, "Light Blue": 0, "Pink": 0, "Orange": 0,
        "Red": 0, "Yellow": 0, "Green": 0, "Dark Blue": 0
    }
    for prop in player_properties:
        if prop.property_set in counts:
            counts[prop.property_set] += 1

    full_sets = []
    requirements = {
        "Brown": 2, "Light Blue": 3, "Pink": 3, "Orange": 3,
        "Red": 3, "Yellow": 3, "Green": 3, "Dark Blue": 2
    }
    for color, required in requirements.items():
        if counts[color] == required:
            full_sets.append(color)

    for prop in player_properties:
        if prop.property_set in full_sets:
            prop.has_full_set = True

    return full_sets


def check_player_solvency(current_player, bank, players):
    """
    Attempt to restore solvency or declare bankruptcy.
    
    Strategy:
    1. Sell hotels first (highest value)
    2. Sell houses (moderate value)
    3. Mortgage properties (last resort)
    
    Returns
    -------
    int
        Total reward from liquidation, or Reward.BANKRUPTCY if bankrupt
    """
    reward = 0
    
    if not current_player.is_bankrupt():
        return reward  # Player is solvent, no action needed
    
    log(MESSAGES.PLAYER_INSOLVENT.format(player_name=current_player.player_name, capital=current_player.capital))
    
    max_iterations = 100  # Safety limit to prevent infinite loops
    iteration = 0
    
    while current_player.capital < 0 and len(current_player.show_assets()) > 0 and iteration < max_iterations:
        progress = False
        iteration += 1
        
        for asset in list(current_player.show_assets()):
            # Stop if solvent
            if current_player.capital >= 0:
                break
            
            if isinstance(asset, Property):
                # Step 1: Sell hotel if exists
                if asset.hotel and current_player.capital < 0:
                    reward += asset.sell_hotel(current_player)
                    sell_price = asset.hotel_price // 2  # Sell at half price
                    current_player.sell(sell_price)
                    bank.sell(sell_price)
                    progress = True
                    log(MESSAGES.SOLD_HOTEL.format(property_name=asset.property_name, sell_price=sell_price))
                
                # Step 2: Sell houses one at a time
                if asset.houses > 0 and current_player.capital < 0:
                    # Sell ONE house per iteration to check solvency
                    reward += asset.sell_house(current_player)
                    sell_price = asset.house_price // 2  # Sell at half price
                    current_player.sell(sell_price)
                    bank.sell(sell_price)
                    progress = True
                    log(MESSAGES.SOLD_HOUSE.format(property_name=asset.property_name, sell_price=sell_price))
                
                # Step 3: Mortgage property if still insolvent
                if not asset.is_mortgaged and current_player.capital < 0:
                    reward += asset.mortgage_property(current_player)
                    mortgage_value = asset.mortgage_value
                    current_player.mortgage(mortgage_value)
                    bank.buy(mortgage_value)
                    progress = True
                    log(MESSAGES.MORTGAGED_ASSET.format(property_name=asset.property_name, mortgage_value=mortgage_value))
            
            elif isinstance(asset, (Railway, Utility)):
                # Mortgage railways and utilities
                if not asset.is_mortgaged and current_player.capital < 0:
                    reward += asset.mortgage_property(current_player)
                    mortgage_value = asset.mortgage_value
                    current_player.mortgage(mortgage_value)
                    bank.mortgage(mortgage_value)
                    progress = True
                    log(MESSAGES.MORTGAGED_ASSET.format(property_name=asset.property_name, mortgage_value=mortgage_value))
                    
        # If no progress made, break to avoid infinite loop
        if not progress:
            log(MESSAGES.NO_MORE_ASSETS)
            break
    
    # Check if safety limit was hit
    if iteration >= max_iterations:
        log(MESSAGES.MAX_ITERATIONS_WARNING)
    
    # Final solvency check
    if current_player.capital < 0:
        log(MESSAGES.PLAYER_BANKRUPT.format(player_name=current_player.player_name, capital=current_player.capital))
        
        # Reset all assets back to the bank
        for asset in list(current_player.show_assets()):
            asset.owner = None
            asset.sold = False
            asset.has_full_set = False
            if isinstance(asset, Property):
                asset.houses = 0
                asset.hotel = False
            if hasattr(asset, 'is_mortgaged'):
                asset.is_mortgaged = False
        
        # Remove player from game
        players.remove(current_player)
        return Reward.BANKRUPTCY
    else:
        log(MESSAGES.PLAYER_RESTORED.format(player_name=current_player.player_name, capital=current_player.capital))
    
    return reward


# ===============================
# MonopolyEnv with Action Masking
# ===============================
class MonopolyEnv(gym.Env):
    def __init__(self, agent_index: int = 0, num_players: int = 4, seed: Optional[int] = None, max_turns: int = 500, use_timer: bool = False, verbose=False):
        super().__init__()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.num_players = num_players
        self.agent_index = agent_index
        self.max_turns = max_turns
        self.turn_count = 0

        # Game components
        self.board = Board()
        self.bank = Bank(salary=200)
        self.trading = Trading_floor()
        self.chance = ChanceDeck()
        self.community = CommunityChestDeck()
        self.verbose = verbose

        # Players
        self.players = [Player(f"Player{i+1}", 1500) for i in range(self.num_players)]
        self.current_player_index = 0

        # Timer and trading/auction bookkeeping (initialized to safe defaults)
        self.use_timer = use_timer
        self.timer: Optional[Timer90] = None
        self.pending_trade = None
        self.active_auction = None
        self.timer_expired = False

        # Compute observation dimensions
        board_info = self.board.board_observation()  
        board_vec = flatten_board_obs(board_info, self.board)
        player_vec = flatten_player_obs(self.board, self.players[0])
        
        self.board_vec_len = board_vec.shape[0]
        self.player_vec_len = player_vec.shape[0]
        self.turn_info_len = 4
        self.action_mask_len = 11

        self.observation_length = self.board_vec_len + self.player_vec_len + self.turn_info_len + self.action_mask_len

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=2.0, 
            shape=(self.observation_length,), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(11)

        # Bookkeeping
        self._prev_net_worth = None
        self._last_turn_info: Optional[Dict[str, Any]] = {
            "dice_roll": 0, "old_position": 0, "new_position": 0, "capital": 1500
        }

        self.logger = MonopolyCSVLogger(env_id=agent_index)

    # -----------------------
    # Action mask generation
    # -----------------------
    def _get_action_mask(self, player: Player) -> np.ndarray:
        """
        Generate 11-action mask (removed ACCEPT_TRADE, DECLINE_TRADE, BID_10, BID_50, BID_100)
        
        New action mapping:
        0: BUY_ASSET
        1: TRADE (auto-resolves)
        2: AUCTION (auto-resolves)
        3: BUILD_HOUSE
        4: BUILD_HOTEL
        5: SELL_HOUSE
        6: SELL_HOTEL
        7: MORTGAGE_ASSET
        8: UNMORTGAGE_ASSET
        9: PAY_TO_LEAVE_JAIL
        10: STAY_IN_JAIL
        """
        mask = np.zeros(11, dtype=np.float32)  # Changed from 16 to 11
        tile = self.board.board[player.position]

        # BUY_ASSET (0)
        if tile.type in ["Property", "Railway", "Utility"] and tile.obj:
            if not tile.obj.is_sold() and player.capital >= tile.obj.market_price:
                mask[0] = 1.0  # Changed from Action.BUY_ASSET to index 0

        # TRADE (1)
        if len(player.show_assets()) > 0 and len(self.players) > 1:
            mask[1] = 1.0

        # AUCTION (2)
        if tile.type in ["Property", "Railway", "Utility"] and tile.obj:
            if not tile.obj.is_sold():
                mask[2] = 1.0

        # BUILD_HOUSE (3)
        can_build_house = False
        for a in player.show_assets():
            if isinstance(a, Property) and a.has_full_set and not a.is_mortgaged and a.houses < 4 and not a.hotel:
                if player.capital >= a.house_price:
                    can_build_house = True
                    break
        mask[3] = 1.0 if can_build_house else 0.0

        # BUILD_HOTEL (4)
        can_build_hotel = False
        for a in player.show_assets():
            if isinstance(a, Property) and a.has_full_set and not a.is_mortgaged and a.houses == 4 and not a.hotel:
                if player.capital >= a.hotel_price:
                    can_build_hotel = True
                    break
        mask[4] = 1.0 if can_build_hotel else 0.0

        # SELL_HOUSE (5)
        can_sell_house = False
        for a in player.show_assets():
            if isinstance(a, Property) and a.houses > 0:
                can_sell_house = True
                break
        mask[5] = 1.0 if can_sell_house else 0.0

        # SELL_HOTEL (6)
        can_sell_hotel = False
        for a in player.show_assets():
            if isinstance(a, Property) and a.hotel:
                can_sell_hotel = True
                break
        mask[6] = 1.0 if can_sell_hotel else 0.0

        # MORTGAGE_ASSET (7)
        can_mortgage = False
        for a in player.show_assets():
            if not getattr(a, "is_mortgaged", False):
                can_mortgage = True
                break
        mask[7] = 1.0 if can_mortgage else 0.0

        # UNMORTGAGE_ASSET (8)
        can_unmortgage = False
        for a in player.show_assets():
            if getattr(a, "is_mortgaged", False):
                unmortgage_cost = getattr(a, "mortgage_value", 0) * 1.1
                if player.capital >= unmortgage_cost:
                    can_unmortgage = True
                    break
        mask[8] = 1.0 if can_unmortgage else 0.0

        # PAY_TO_LEAVE_JAIL (9)
        if player.in_jail and player.capital >= 50:
            mask[9] = 1.0

        # STAY_IN_JAIL (10)
        if player.in_jail:
            mask[10] = 1.0

        return mask

    # -----------------------
    # Observation builder
    # -----------------------
    def _get_observation(self) -> np.ndarray:
        """Build concatenated observation: [board_vec | player_vec | turn_info_vec | action_mask]"""
        agent_player = self.players[self.agent_index]
        
        board_info = self.board.board_observation()
        board_vec = flatten_board_obs(board_info, self.board)
        player_vec = flatten_player_obs(self.board, agent_player)
        turn_vec = flatten_turn_info(self._last_turn_info)
        action_mask = self._get_action_mask(agent_player)

        obs = np.concatenate([board_vec, player_vec, turn_vec, action_mask]).astype(np.float32)
        return obs

    # -----------------------
    # Reset
    # -----------------------
    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.logger.start_episode()
        self.turn_count = 0
        self.board = Board()
        self.bank = Bank(salary=200)
        self.trading = Trading_floor()
        self.chance = ChanceDeck()
        self.community = CommunityChestDeck()

        self.players = [Player(f"Player{i+1}", 1500) for i in range(self.num_players)]
        self.current_player_index = 0
        
        # Reset auction and trade state
        self.active_auction = None
        self.pending_trade = None
        
        # Reset timer
        if self.timer:
            try:
                self.timer.stop()
            except Exception:
                pass
            self.timer = Timer90()
        else:
            # ensure timer exists if use_timer True
            if self.use_timer:
                self.timer = Timer90()
        self.timer_expired = False

        agent = self.players[self.agent_index]
        self._prev_net_worth = agent.capital + getattr(agent, "net_asset_value", 0)
        self._last_turn_info = {
            "dice_roll": 0, "old_position": 0, "new_position": agent.position, "capital": agent.capital
        }

        obs = self._get_observation()
        return obs, {}

    # -----------------------
    # Step
    # -----------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.turn_count += 1
        self.logger.step()

        # -------------------------
        # Timer handling
        # -------------------------
        if self.use_timer and self.current_player_index == self.agent_index:
            if hasattr(self, "timer") and self.timer:
                try:
                    self.timer.stop()
                except Exception:
                    pass
            self.timer = Timer90()
            try:
                self.timer.start()
            except Exception:
                pass
            self.timer_expired = False

        # -------------------------
        # Fast-forward heuristic players
        # -------------------------
        while self.current_player_index != self.agent_index:
            current = self.players[self.current_player_index]
            self._run_heuristic_turn(current)
            self.current_player_index = (self.current_player_index + 1) % len(self.players)

            # Agent eliminated during heuristic turns
            if self.agent_index >= len(self.players):
                if hasattr(self, "timer") and self.timer:
                    try:
                        self.timer.stop()
                    except Exception:
                        pass
                obs = self._get_observation()
                return obs, float(Reward.BANKRUPTCY), True, False, {"reason": "agent_removed"}

        # -------------------------
        # Agent's turn
        # -------------------------
        agent = self.players[self.agent_index]
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        # -------------------------
        # Timer expiration
        # -------------------------
        if self.use_timer and hasattr(self, "timer") and self.timer and getattr(self.timer, "time_left", 1) <= 0:
            self.timer_expired = True
            reward += Reward.TIMEOUT
            info["timer_expired"] = True

        # -------------------------
        # Movement phase
        # -------------------------
        dice = roll_dice()
        old_pos = agent.position
        new_pos = (old_pos + dice) % 40
        agent.position = new_pos

        self._last_turn_info = {
            "dice_roll": dice,
            "old_position": old_pos,
            "new_position": new_pos,
            "capital": agent.capital,
        }

        # Passing GO
        if old_pos + dice >= 40:
            salary = self.bank.go_salary()
            agent.receive(salary)

        # Tile interaction
        tile = self.board.board[new_pos]
        tile_interaction(agent, tile, self.bank, self.board, self.players, dice, self.chance, self.community)

        # Full-set reward
        if check_set_ownership(agent):
            reward += Reward.FULL_SET

        # -------------------------
        # Action phase
        # -------------------------
        if self.timer_expired:
            action = -1  # forced no-op

        action_mask = self._get_action_mask(agent)
        invalid_action = action >= 0 and action_mask[action] == 0

        if invalid_action:
            reward += Reward.INVALID_ACTION
        elif action >= 0:
            reward += self._execute_action(agent, action)

        # -------------------------
        # Solvency check
        # -------------------------
        bankruptcy_penalty = check_player_solvency(agent, self.bank, self.players)
        if bankruptcy_penalty:
            reward += bankruptcy_penalty
            done = True
            info["reason"] = "bankruptcy"

        # -------------------------
        # Net worth
        # -------------------------
        net_worth = agent.capital + getattr(agent, "net_asset_value", 0)
        self._prev_net_worth = net_worth

        if not self.bank.bank_solvency:
            done = True
            info["reason"] = "bank_insolvent"

        # -------------------------
        # Win condition
        # -------------------------
        if len(self.players) == 1:
            done = True
            if self.players[0] == agent:
                reward += Reward.WIN_GAME
                info["reason"] = "victory"

        # -------------------------
        # Max turn truncation
        # -------------------------
        if self.turn_count >= self.max_turns:
            done = True
            info["reason"] = "max_turns"

        # -------------------------
        # Stop timer
        # -------------------------
        if self.use_timer and hasattr(self, "timer") and self.timer:
            try:
                self.timer.stop()
            except Exception:
                pass

        # -------------------------
        # Advance turn
        # -------------------------
        if len(self.players) > 0:
            self.current_player_index = (self.current_player_index + 1) % len(self.players)

        # -------------------------
        # LOGGING (CRITICAL)
        # -------------------------
        winner_index = None
        if done and len(self.players) == 1:
            winner_index = self.players.index(self.players[0])

        self.logger.log_step(
            players=self.players,
            agent_index=self.agent_index,
            action=action,
            reward=reward,
            action_mask=action_mask,
            invalid_action=invalid_action,
            done=done,
            winner_index=winner_index,
        )

        # -------------------------
        # Return
        # -------------------------
        obs = self._get_observation()
        info["net_worth"] = net_worth
        info["turn"] = self.turn_count
        if self.use_timer and hasattr(self, "timer") and self.timer:
            info["time_left"] = getattr(self.timer, "time_left", None)

        return obs, float(reward), done, False, info



    # -----------------------
    # Action execution
    # -----------------------
    def _execute_action(self, agent: Player, action: int) -> float:
        """Execute action using new 11-action mapping."""
        reward = 0.0

        if action == 0:  # BUY_ASSET
            tile = self.board.board[agent.position]
            if tile.type in ["Property", "Railway", "Utility"] and tile.obj and not tile.obj.is_sold():
                price = tile.obj.market_price
                if agent.capital >= price:
                    agent.buy(tile.obj, price)
                    tile.obj.buy(agent)
                    self.bank.buy(price)
                    reward = Reward.BUY_ASSET

        elif action == 1:  # TRADE
            reward = self._initiate_trade(agent)

        elif action == 2:  # AUCTION
            tile = self.board.board[agent.position]
            if tile.type in ["Property", "Railway", "Utility"] and tile.obj and not tile.obj.is_sold():
                self._start_auction(tile.obj)
                reward = 0.0

        elif action == 3:  # BUILD_HOUSE
            for a in list(agent.assets):
                if isinstance(a, Property) and a.has_full_set and not a.is_mortgaged and a.houses < 4 and not a.hotel:
                    cost = a.house_price
                    if agent.capital >= cost:
                        agent.build(a, cost)
                        a.build_house(agent)
                        self.bank.buy(cost)
                        reward = Reward.BUILD_HOUSE
                        break

        elif action == 4:  # BUILD_HOTEL
            for a in list(agent.assets):
                if isinstance(a, Property) and a.has_full_set and not a.is_mortgaged and a.houses == 4 and not a.hotel:
                    cost = a.hotel_price
                    if agent.capital >= cost:
                        agent.build(a, cost)
                        a.build_hotel(agent)
                        self.bank.buy(cost)
                        reward = Reward.BUILD_HOTEL
                        break

        elif action == 5:  # SELL_HOUSE
            for a in list(agent.assets):
                if isinstance(a, Property) and a.houses > 0:
                    price = a.house_price // 2
                    a.sell_house(agent)
                    agent.sell(price)
                    self.bank.sell(price)
                    reward = Reward.SELL_HOUSE
                    break

        elif action == 6:  # SELL_HOTEL
            for a in list(agent.assets):
                if isinstance(a, Property) and a.hotel:
                    price = a.hotel_price // 2
                    a.sell_hotel(agent)
                    agent.sell(price)
                    self.bank.sell(price)
                    reward = Reward.SELL_HOTEL
                    break

        elif action == 7:  # MORTGAGE_ASSET
            for a in list(agent.assets):
                if not getattr(a, "is_mortgaged", False):
                    val = a.mortgage_property(agent)
                    if isinstance(val, (int, float)):
                        agent.mortgage(val)
                        self.bank.mortgage(val)
                        reward = Reward.MORTGAGE_ASSET
                        break

        elif action == 8:  # UNMORTGAGE_ASSET
            for a in list(agent.assets):
                if getattr(a, "is_mortgaged", False):
                    unmortgage_cost = getattr(a, "mortgage_value", 0) * 1.1
                    if agent.capital >= unmortgage_cost:
                        val = a.unmortgage_property(agent)
                        if isinstance(val, (int, float)):
                            agent.unmortgage(unmortgage_cost)
                            self.bank.unmortgage(unmortgage_cost)
                            reward = Reward.UNMORTGAGE_ASSET
                            break

        elif action == 9:  # PAY_TO_LEAVE_JAIL
            if agent.in_jail and agent.capital >= 50:
                agent.pay(50)
                agent.in_jail = False
                agent.jail_counter = 0
                self.bank.buy(50)
                reward = Reward.PAY_TO_LEAVE

        elif action == 10:  # STAY_IN_JAIL
            if agent.in_jail:
                agent.jail_counter += 1
                if agent.jail_counter >= 3:
                    agent.in_jail = False
                    agent.jail_counter = 0
                reward = Reward.STAY

        return reward


    # -----------------------
    # Auction helpers
    # -----------------------
    def _start_auction(self, property_obj):
        """Initialize and IMMEDIATELY RESOLVE auction."""
        # DON'T create pending auction state - resolve it immediately
        # This prevents the environment from getting stuck waiting for bids
        
        auction = Auction(property_obj, self.players.copy())
        winner = auction.start_auction(self.bank)
        
        if winner:
            log(MESSAGES.WINS_AUCTION.format(player_name=winner.player_name, property_name=property_obj.property_name, final_bid=property_obj.market_price))
    
        # No persistent auction state
        self.active_auction = None

    # -----------------------
    # Trading helpers
    # -----------------------
    def _initiate_trade(self, proposer: Player) -> float:
        """SIMPLIFIED: Auto-evaluate and execute trade immediately."""
        
        other_players = [p for p in self.players if p != proposer]
        if not other_players:
            return Reward.NO_REWARD

        receiver = random.choice(other_players)
        proposer_assets = proposer.show_assets()
        if not proposer_assets:
            return Reward.NO_REWARD
            
        asset1 = random.choice(proposer_assets)
        receiver_assets = receiver.show_assets()
        asset2 = random.choice(receiver_assets) if receiver_assets else None
        
        pay_amount = 0
        ask_amount = int(asset1.market_price * 0.8) if asset2 is None else 0
        
        # IMMEDIATELY EVALUATE using heuristic
        proposal = self.trading.proposition(asset1, asset2, pay_amount, ask_amount)
        
        # Use receiver's method if available, otherwise heuristic
        if hasattr(receiver, 'see_proposition') and callable(receiver.see_proposition):
            accept = receiver.see_proposition(proposal)
        else:
            accept = self._heuristic_trade_decision(receiver, asset1, asset2, pay_amount, ask_amount)
        
        # IMMEDIATELY EXECUTE if accepted
        if accept:
            try:
                result = self.trading.open_trade(proposer, receiver, asset1, asset2, pay_amount, ask_amount)
                if "successful" in result:
                    log(MESSAGES.TRADE_SUCCESSFUL.format(proposer_name=proposer.player_name, receiver_name=receiver.player_name))
                    return Reward.NO_REWARD  # Reward for successful trade
            except Exception as e:
                log(MESSAGES.TRADE_FAILED.format(error_msg=str(e)))
                return Reward.NO_REWARD
        
        # Trade rejected
        return Reward.NO_REWARD

    # -----------------------
    # Heuristic player logic
    # -----------------------
    def _run_heuristic_turn(self, player: Player):
        """Simple heuristic for non-agent players."""
        dice = random.randint(1, 6)
        old_pos = player.position
        player.position = (old_pos + dice) % 40

        if old_pos + dice >= 40:
            salary = self.bank.go_salary()
            player.receive(salary)

        tile = self.board.board[player.position]
        
        # Handle pending trade if player is receiver
        if self.pending_trade is not None:
            proposer, receiver, asset1, asset2, pay_amount, ask_amount = self.pending_trade
            if receiver == player:
                # Use player's see_proposition if available
                if hasattr(player, 'see_proposition') and callable(player.see_proposition):
                    proposal = self.trading.proposition(asset1, asset2, pay_amount, ask_amount)
                    accept = player.see_proposition(proposal)
                else:
                    # Use heuristic
                    accept = self._heuristic_trade_decision(receiver, asset1, asset2, pay_amount, ask_amount)
                
                if accept:
                    # Use self.trading (your Trading_floor instance)
                    try:
                        self.trading.open_trade(proposer, receiver, asset1, asset2, pay_amount, ask_amount)
                    except Exception:
                        pass
                self.pending_trade = None
        
        # Handle active auction (basic)
        # (left as-is; more robust auction handling can be added)
        
        # Auto-buy if affordable and no auction active
        if tile.type in ["Property", "Railway", "Utility"] and tile.obj and not tile.obj.is_sold():
            if self.active_auction is None and player.capital >= tile.obj.market_price * 1.2:
                # Only buy if we have 20% buffer
                player.buy(tile.obj, tile.obj.market_price)
                tile.obj.buy(player)
                self.bank.buy(tile.obj.market_price)

        # Check for full sets and build
        full_sets = check_set_ownership(player)
        if full_sets:
            for a in list(player.assets):
                if isinstance(a, Property) and a.has_full_set and not a.is_mortgaged:
                    # Build house
                    if a.houses < 4 and not a.hotel and player.capital >= a.house_price * 2:
                        player.build(a, a.house_price)
                        a.build_house(player)
                        self.bank.buy(a.house_price)
                    # Build hotel
                    elif a.houses == 4 and not a.hotel and player.capital >= a.hotel_price * 2:
                        player.build(a, a.hotel_price)
                        a.build_hotel(player)
                        self.bank.buy(a.hotel_price)

        # Check solvency
        check_player_solvency(player, self.bank, self.players)

    def _heuristic_trade_decision(self, receiver: Player, asset1, asset2, pay_amount: int, ask_amount: int) -> bool:
        """Heuristic decision for accepting/declining trades."""
        # Simple rule: accept if we're getting an asset and not giving much
        if asset1 is not None and asset2 is None and ask_amount < asset1.market_price * 0.7:
            return True
        
        # Accept if we're getting better value
        giving_value = (asset2.market_price if asset2 else 0) + ask_amount
        receiving_value = (asset1.market_price if asset1 else 0) + pay_amount
        
        return receiving_value > giving_value * 1.1

    def _heuristic_auction_bid(self, player: Player, property_obj, current_bid: int) -> Optional[int]:
        """Heuristic for auction bidding.""" 
        max_bid = min(property_obj.market_price * 0.8, player.capital * 0.3)
        
        if current_bid >= max_bid:
            return None  # Drop out
        
        # Bid in increments
        if current_bid < 50:
            return current_bid + 10
        elif current_bid < 200:
            return current_bid + 50
        else:
            return current_bid + 100

    # ---------------------------
    # Rendering
    # ---------------------------
    def render(self, mode: str = "human"):
        """Render the current game state."""
        agent = self.players[self.agent_index] if self.agent_index < len(self.players) else None
        print("\n" + "="*60)
        print("MONOPOLY GAME STATE")
        print("="*60)
        
        # Player information
        for i, p in enumerate(self.players):
            flag = "AGENT" if i == self.agent_index else "PLAYER"
            jail_status = "IN JAIL" if p.in_jail else ""
            print(f"{p.player_name:12s} {flag:10s} | Cash: ${p.capital:7.0f} | Pos: {p.position:2d} | Assets: {len(p.assets):2d} {jail_status}")
        
        print("-"*60)
        
        # Agent details
        if agent:
            net_worth = agent.capital + getattr(agent, 'net_asset_value', 0)
            print(f"Agent Net Worth: ${net_worth:8.0f}")
            
            if len(agent.assets) > 0:
                print("\nAgent Assets:")
                for asset in agent.assets:
                    asset_type = "Property" if isinstance(asset, Property) else "Railway" if isinstance(asset, Railway) else "Utility"
                    mortgaged = " [MORTGAGED]" if getattr(asset, "is_mortgaged", False) else ""
                    houses = f" ({asset.houses}H)" if isinstance(asset, Property) and asset.houses > 0 else ""
                    hotel = " (HOTEL)" if isinstance(asset, Property) and asset.hotel else ""
                    print(f"  {asset_type} {asset.property_name:30s} ${asset.market_price:4.0f}{mortgaged}{houses}{hotel}")
        
        # Game state
        print("-"*60)
        print(f"Turn: {self.turn_count}/{self.max_turns}")
        if self.use_timer and self.timer:
            print(f"Time Left: {getattr(self.timer, 'time_left', None)}s")
        if self.pending_trade:
            print("Trade Pending")
        if self.active_auction:
            # active_auction may be a dict or object; handle generically
            prop_name = None
            try:
                prop_name = self.active_auction.get("property").property_name
            except Exception:
                try:
                    prop_name = getattr(self.active_auction, "property_name", None)
                except Exception:
                    prop_name = None
            print(f"Auction Active: {prop_name}")
        
        print("="*60 + "\n")

    def close(self):
        """Clean up resources including timer."""
        if getattr(self, "timer", None):
            try:
                self.timer.stop()
            except Exception:
                pass
