# monopoly_env.py
"""
MonopolyEnv - Gymnasium-compatible environment for Monopoly with proper action masking.

Key improvements:
- Action masking to prevent invalid actions
- Better reward shaping with net worth tracking
- Cleaner separation of agent and heuristic player logic
- Proper handling of all 16 actions with Auction and Trading_floor classes
- Episode termination conditions
- 90-second turn timer integration
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
from trading_floor import Trading_floor
from player import *   
from auction_floor import Auction
from cards import *
from rewards import *
from actions import *
from turn_timer import Timer90

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
        case "move_back":
            player.position = (player.position - card.amount) % 40
        case "jail":
            player.position = 10
            if player.has_jail_free_card:
                player.has_jail_free_card = False
            else:
                player.in_jail = True
        case "pay_bank":
            player.pay(card.amount)
            bank.buy(card.amount)
        case "gain_bank":
            player.receive(card.amount)
            bank.sell(card.amount)
        case "jail_free":
            player.has_jail_free_card = True
        case "repair":
            house_cost = card.target["house"]
            hotel_cost = card.target["hotel"]
            total = 0
            for a in player.show_assets():
                if isinstance(a, Property):
                    total += a.houses * house_cost
                    total += a.hotel * hotel_cost
            player.pay(total)
            bank.buy(total)
        case "nearest_railway":
            rail_positions = [5, 15, 25, 35]
            new_pos = min(rail_positions, key=lambda p: (p - player.position) % 40)
            player.position = new_pos
        case "nearest_utility":
            util_positions = [12, 28]
            new_pos = min(util_positions, key=lambda p: (p - player.position) % 40)
            player.position = new_pos


def tile_interaction(current_player, tile, bank, board, players, dice, chance_deck, community_deck):
    """Auto-handle tile landing (for movement phase)."""
    if tile.get_type() == "Property":
        prop = tile.get_obj()
        if prop.sold and prop.owner != current_player:
            rent = prop.pay_rent()
            current_player.pay(rent)
            prop.owner.receive(rent)

    elif tile.get_type() == "Railway":
        rail = tile.get_obj()
        if rail.sold and rail.owner != current_player:
            rent = rail.pay_rent()
            current_player.pay(rent)
            rail.owner.receive(rent)

    elif tile.get_type() == "Utility":
        util = tile.get_obj()
        if util.sold and util.owner != current_player:
            rent = util.pay_rent(dice)
            current_player.pay(rent)
            util.owner.receive(rent)

    elif tile.get_type() == "Special":
        if tile.obj == "Income Tax":
            current_player.pay(200)
            bank.buy(200)
        elif tile.obj == "Luxury Tax":
            current_player.pay(100)
            bank.buy(100)
        elif tile.obj == "Go To Jail":
            current_player.position = 10
            if current_player.has_jail_free_card:
                current_player.has_jail_free_card = False
            else:
                current_player.in_jail = True
        elif tile.obj == "Chance":
            card = chance_deck.draw()
            resolve_card(current_player, card, board, bank, players, dice)
        elif tile.obj == "Community Chest":
            card = community_deck.draw()
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
    """Attempt to restore solvency or declare bankruptcy."""
    if current_player.is_bankrupt():
        while current_player.capital < 0 and len(current_player.show_assets()) > 0:
            progress = False

            for asset in list(current_player.show_assets()):
                if isinstance(asset, Property):
                    if asset.hotel and current_player.capital < 0:
                        sell_price = asset.sell_hotel(current_player)
                        current_player.sell(sell_price)
                        bank.sell(sell_price)
                        progress = True

                    while asset.houses > 0 and current_player.capital < 0:
                        sell_price = asset.sell_house(current_player)
                        current_player.sell(sell_price)
                        bank.sell(sell_price)
                        progress = True

                    if not asset.is_mortgaged and current_player.capital < 0:
                        mortgage_value = asset.mortgage_property(current_player)
                        current_player.mortgage(mortgage_value)
                        bank.buy(mortgage_value)
                        progress = True

                elif isinstance(asset, (Railway, Utility)):
                    if not asset.is_mortgaged and current_player.capital < 0:
                        sell_price = asset.mortgage_property(current_player)
                        current_player.mortgage(sell_price)
                        bank.mortgage(sell_price)
                        progress = True

            if not progress:
                break

        if current_player.capital < 0:
            print(f"{current_player.player_name} is bankrupt!")

            # Reset assets
            for asset in current_player.show_assets():
                asset.owner = None
                asset.sold = False
                asset.has_full_set = False
                asset.houses = 0
                asset.hotel = False

            players.remove(current_player)
            return Reward.BANKRUPTCY
    return 0


# ===============================
# MonopolyEnv with Action Masking
# ===============================
class MonopolyEnv(gym.Env):
    """
    Gymnasium Monopoly environment with:
    - Vector observations
    - Action masking (16 actions)
    - Reward shaping based on net worth
    - Full auction and trading support using Auction and Trading_floor classes
    - 90-second turn timer
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, agent_index: int = 0, num_players: int = 4, seed: Optional[int] = None, max_turns: int = 500, use_timer: bool = False):
        super().__init__()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.num_players = num_players
        self.agent_index = agent_index
        self.max_turns = max_turns
        self.turn_count = 0
        self.use_timer = use_timer

        # Game components
        self.board = Board()
        self.bank = Bank(salary=200)
        self.trading_floor = Trading_floor()
        self.chance = ChanceDeck()
        self.community = CommunityChestDeck()

        # Players
        self.players = [Player(f"Player{i+1}", 1500) for i in range(self.num_players)]
        self.current_player_index = 0
        
        # Auction and trading state
        self.active_auction = None  # Auction object
        self.pending_trade = None  # (proposer, receiver, asset1, asset2, pay_amount, ask_amount)
        
        # Timer
        self.timer = Timer90() if use_timer else None
        self.timer_expired = False

        # Compute observation dimensions
        board_info = self.board.board_observation()  
        board_vec = flatten_board_obs(board_info, self.board)
        player_vec = flatten_player_obs(self.board, self.players[0])
        
        self.board_vec_len = board_vec.shape[0]
        self.player_vec_len = player_vec.shape[0]
        self.turn_info_len = 4
        self.action_mask_len = 16

        self.observation_length = self.board_vec_len + self.player_vec_len + self.turn_info_len + self.action_mask_len

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=2.0, 
            shape=(self.observation_length,), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(16)

        # Bookkeeping
        self._prev_net_worth = None
        self._last_turn_info: Optional[Dict[str, Any]] = {
            "dice_roll": 0, "old_position": 0, "new_position": 0, "capital": 1500
        }

    # -----------------------
    # Action mask generation
    # -----------------------
    def _get_action_mask(self, player: Player) -> np.ndarray:
        """
        Generate binary mask (0/1) for each of the 16 actions.
        1 = action is valid, 0 = action is invalid.
        """
        mask = np.zeros(16, dtype=np.float32)
        tile = self.board.board[player.position]

        # BUY_ASSET (0)
        if tile.type in ["Property", "Railway", "Utility"] and tile.obj:
            if not tile.obj.is_sold() and player.capital >= tile.obj.market_price:
                mask[Action.BUY_ASSET] = 1.0

        # TRADE (1) - can propose if player owns assets and there are other players
        if len(player.show_assets()) > 0 and len(self.players) > 1:
            mask[Action.TRADE] = 1.0

        # ACCEPT_TRADE (2) / DECLINE_TRADE (3) - only if trade pending for this player
        if self.pending_trade is not None:
            proposer, receiver, _, _, _, _ = self.pending_trade
            if receiver == player:
                mask[Action.ACCEPT_TRADE] = 1.0
                mask[Action.DECLINE_TRADE] = 1.0

        # AUCTION (4) - can start auction if on unowned property and no active auction
        if tile.type in ["Property", "Railway", "Utility"] and tile.obj:
            if not tile.obj.is_sold() and self.active_auction is None:
                mask[Action.AUCTION] = 1.0

        # BID actions (5-7) - only during active auction
        if self.active_auction is not None:
            if player.capital >= 10:
                mask[Action.BID_10] = 1.0
            if player.capital >= 50:
                mask[Action.BID_50] = 1.0
            if player.capital >= 100:
                mask[Action.BID_100] = 1.0

        # BUILD_HOUSE (8)
        can_build_house = False
        for a in player.show_assets():
            if isinstance(a, Property) and a.has_full_set and not a.is_mortgaged and a.houses < 4 and not a.hotel:
                if player.capital >= a.house_price:
                    can_build_house = True
                    break
        mask[Action.BUILD_HOUSE] = 1.0 if can_build_house else 0.0

        # BUILD_HOTEL (9)
        can_build_hotel = False
        for a in player.show_assets():
            if isinstance(a, Property) and a.has_full_set and not a.is_mortgaged and a.houses == 4 and not a.hotel:
                if player.capital >= a.hotel_price:
                    can_build_hotel = True
                    break
        mask[Action.BUILD_HOTEL] = 1.0 if can_build_hotel else 0.0

        # SELL_HOUSE (10)
        can_sell_house = False
        for a in player.show_assets():
            if isinstance(a, Property) and a.houses > 0:
                can_sell_house = True
                break
        mask[Action.SELL_HOUSE] = 1.0 if can_sell_house else 0.0

        # SELL_HOTEL (11)
        can_sell_hotel = False
        for a in player.show_assets():
            if isinstance(a, Property) and a.hotel:
                can_sell_hotel = True
                break
        mask[Action.SELL_HOTEL] = 1.0 if can_sell_hotel else 0.0

        # MORTGAGE_ASSET (12)
        can_mortgage = False
        for a in player.show_assets():
            if not getattr(a, "is_mortgaged", False):
                can_mortgage = True
                break
        mask[Action.MORTGAGE_ASSET] = 1.0 if can_mortgage else 0.0

        # UNMORTGAGE_ASSET (13)
        can_unmortgage = False
        for a in player.show_assets():
            if getattr(a, "is_mortgaged", False):
                unmortgage_cost = getattr(a, "mortgage_value", 0) * 1.1
                if player.capital >= unmortgage_cost:
                    can_unmortgage = True
                    break
        mask[Action.UNMORTGAGE_ASSET] = 1.0 if can_unmortgage else 0.0

        # PAY_TO_LEAVE_JAIL (14)
        if player.in_jail and player.capital >= 50:
            mask[Action.PAY_TO_LEAVE_JAIL] = 1.0

        # STAY_IN_JAIL (15)
        if player.in_jail:
            mask[Action.STAY_IN_JAIL] = 1.0

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

        self.turn_count = 0
        self.board = Board()
        self.bank = Bank(salary=200)
        self.trading_floor = Trading_floor()
        self.chance = ChanceDeck()
        self.community = CommunityChestDeck()

        self.players = [Player(f"Player{i+1}", 1500) for i in range(self.num_players)]
        self.current_player_index = 0
        
        # Reset auction and trade state
        self.active_auction = None
        self.pending_trade = None
        
        # Reset timer
        if self.timer:
            self.timer.stop()
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
        
        # Start timer for agent's turn
        if self.use_timer and self.current_player_index == self.agent_index:
            if self.timer:
                self.timer.stop()
            self.timer = Timer90()
            self.timer.start()
            self.timer_expired = False

        # Fast-forward heuristic players until agent's turn
        while self.current_player_index != self.agent_index:
            current = self.players[self.current_player_index]
            self._run_heuristic_turn(current)
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
            
            if self.agent_index >= len(self.players):
                if self.timer:
                    self.timer.stop()
                obs = self._get_observation()
                return obs, float(Reward.BANKRUPTCY), True, False, {"reason": "agent_removed"}

        # Agent's turn
        agent = self.players[self.agent_index]
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        
        # Check if timer expired (only applicable if timer is enabled)
        if self.use_timer and self.timer and self.timer.time_left <= 0:
            self.timer_expired = True
            reward -= 2.0  # Penalty for timing out
            info["timer_expired"] = True
            print(f"\n{agent.player_name}'s turn timed out!")

        # ---- Movement Phase ----
        dice = roll_dice()
        old_pos = agent.position
        new_pos = (old_pos + dice) % 40
        agent.position = new_pos

        self._last_turn_info = {
            "dice_roll": dice, "old_position": old_pos, "new_position": new_pos, "capital": agent.capital
        }

        # Passing GO
        if old_pos + dice >= 40:
            salary = self.bank.go_salary()
            agent.receive(salary)
            reward += 0.1

        # Tile interaction (rent, taxes, cards)
        tile = self.board.board[new_pos]
        tile_interaction(agent, tile, self.bank, self.board, self.players, dice, self.chance, self.community)

        # Check full sets after movement
        full_sets = check_set_ownership(agent)
        if full_sets:
            reward += Reward.FULL_SET

        # ---- Action Phase ----
        # If timer expired, force a default action (do nothing / pass turn)
        if self.timer_expired:
            action = Action.STAY_IN_JAIL if agent.in_jail else -1  # No-op
        
        action_mask = self._get_action_mask(agent)
        
        # Penalize invalid actions
        if action >= 0 and action_mask[action] == 0.0:
            reward -= 1.0
        elif action >= 0:
            # Execute valid action
            action_reward = self._execute_action(agent, action)
            reward += action_reward

        # ---- Post-action checks ----
        # Solvency check
        bankruptcy_penalty = check_player_solvency(agent, self.bank, self.players)
        if bankruptcy_penalty:
            reward += bankruptcy_penalty
            done = True
            info["reason"] = "bankruptcy"

        # Net worth shaping
        net_worth = agent.capital + getattr(agent, "net_asset_value", 0)
        if self._prev_net_worth is not None:
            delta = net_worth - self._prev_net_worth
            reward += float(delta) / 1000.0
        self._prev_net_worth = net_worth

        # Win condition
        if len(self.players) == 1:
            done = True
            if self.players[0] == agent:
                reward += Reward.WIN_GAME
                info["reason"] = "victory"

        # Max turns truncation
        if self.turn_count >= self.max_turns:
            done = True
            info["reason"] = "max_turns"
        
        # Stop timer when turn ends
        if self.use_timer and self.timer:
            self.timer.stop()

        # Advance turn
        if len(self.players) > 0:
            self.current_player_index = (self.current_player_index + 1) % len(self.players)

        obs = self._get_observation()
        info["net_worth"] = net_worth
        info["turn"] = self.turn_count
        if self.use_timer and self.timer:
            info["time_left"] = self.timer.time_left
        
        return obs, float(reward), done, False, info

    # -----------------------
    # Action execution
    # -----------------------
    def _execute_action(self, agent: Player, action: int) -> float:
        """Execute the chosen action and return immediate reward."""
        reward = 0.0

        if action == Action.BUY_ASSET:
            tile = self.board.board[agent.position]
            if tile.type in ["Property", "Railway", "Utility"] and tile.obj and not tile.obj.is_sold():
                price = tile.obj.market_price
                if agent.capital >= price:
                    agent.buy(tile.obj, price)
                    tile.obj.buy(agent)
                    self.bank.buy(price)
                    reward = Reward.BUY_ASSET

        elif action == Action.TRADE:
            # Propose a trade with a random other player
            reward = self._initiate_trade(agent)

        elif action == Action.ACCEPT_TRADE:
            if self.pending_trade is not None:
                proposer, receiver, asset1, asset2, pay_amount, ask_amount = self.pending_trade
                if receiver == agent:
                    # Use Trading_floor.open_trade
                    result = self.trading_floor.open_trade(proposer, receiver, asset1, asset2, pay_amount, ask_amount)
                    if "successful" in result:
                        reward = 2.0  # Reward for successful trade
                    self.pending_trade = None

        elif action == Action.DECLINE_TRADE:
            if self.pending_trade is not None:
                _, receiver, _, _, _, _ = self.pending_trade
                if receiver == agent:
                    self.pending_trade = None
                    reward = -0.5  # Small penalty for declining

        elif action == Action.AUCTION:
            # Start auction using Auction class
            tile = self.board.board[agent.position]
            if tile.type in ["Property", "Railway", "Utility"] and tile.obj and not tile.obj.is_sold():
                self._start_auction(tile.obj)
                reward = 0.0

        elif action == Action.BID_10:
            reward = self._place_bid(agent, 10)

        elif action == Action.BID_50:
            reward = self._place_bid(agent, 50)

        elif action == Action.BID_100:
            reward = self._place_bid(agent, 100)

        elif action == Action.BUILD_HOUSE:
            for a in list(agent.assets):
                if isinstance(a, Property) and a.has_full_set and not a.is_mortgaged and a.houses < 4 and not a.hotel:
                    cost = a.house_price
                    if agent.capital >= cost:
                        agent.build(a, cost)
                        a.build_house(agent)
                        self.bank.buy(cost)
                        reward = Reward.BUILD_HOUSE
                        break

        elif action == Action.BUILD_HOTEL:
            for a in list(agent.assets):
                if isinstance(a, Property) and a.has_full_set and not a.is_mortgaged and a.houses == 4 and not a.hotel:
                    cost = a.hotel_price
                    if agent.capital >= cost:
                        agent.build(a, cost)
                        a.build_hotel(agent)
                        self.bank.buy(cost)
                        reward = Reward.BUILD_HOTEL
                        break

        elif action == Action.SELL_HOUSE:
            for a in list(agent.assets):
                if isinstance(a, Property) and a.houses > 0:
                    price = a.sell_house(agent)
                    agent.sell(price)
                    self.bank.sell(price)
                    reward = Reward.SELL_HOUSE
                    break

        elif action == Action.SELL_HOTEL:
            for a in list(agent.assets):
                if isinstance(a, Property) and a.hotel:
                    price = a.sell_hotel(agent)
                    agent.sell(price)
                    self.bank.sell(price)
                    reward = Reward.SELL_HOTEL
                    break

        elif action == Action.MORTGAGE_ASSET:
            for a in list(agent.assets):
                if not getattr(a, "is_mortgaged", False):
                    val = a.mortgage_property(agent)
                    if isinstance(val, (int, float)):
                        agent.mortgage(val)
                        self.bank.mortgage(val)
                        reward = Reward.MORTGAGE_ASSET
                        break

        elif action == Action.UNMORTGAGE_ASSET:
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

        elif action == Action.PAY_TO_LEAVE_JAIL:
            if agent.in_jail and agent.capital >= 50:
                agent.pay(50)
                agent.in_jail = False
                agent.jail_counter = 0
                self.bank.buy(50)
                reward = Reward.PAY_TO_LEAVE

        elif action == Action.STAY_IN_JAIL:
            if agent.in_jail:
                agent.jail_counter += 1
                if agent.jail_counter >= 3:
                    agent.in_jail = False
                    agent.jail_counter = 0
                reward = Reward.STAY

        return reward

    # -----------------------
    # Auction helpers using Auction class
    # -----------------------
    def _start_auction(self, property_obj):
        """Initialize an auction using the Auction class."""
        self.active_auction = Auction(property_obj, self.players)
        print(f"Auction started for {property_obj.property_name}")

    def _place_bid(self, player: Player, bid_amount: int) -> float:
        """Handle a bid during an active auction."""
        if self.active_auction is None:
            return -0.5

        if player.capital < bid_amount:
            return -0.5

        # Use player's make_bid method if available, otherwise use simple logic
        if hasattr(player, 'make_bid') and callable(player.make_bid):
            bid = player.make_bid(self.active_auction.property, bid_amount)
            if bid and bid > bid_amount:
                print(f"{player.player_name} bids {bid}")
                return 0.5
        
        # Default bidding behavior
        print(f"{player.player_name} bids {bid_amount}")
        return 0.5

    def _resolve_auction(self):
        """Finalize the auction using the Auction class."""
        if self.active_auction is None:
            return

        # Use the Auction class's start_auction method
        winner = self.active_auction.start_auction(self.bank)
        self.active_auction = None
        
        if winner:
            print(f"Auction won by {winner.player_name}")

    # -----------------------
    # Trading helpers using Trading_floor class
    # -----------------------
    def _initiate_trade(self, proposer: Player) -> float:
        """Agent initiates a trade proposal."""
        # Simple strategy: offer a random asset to a random other player
        other_players = [p for p in self.players if p != proposer]
        if not other_players:
            return -0.5

        receiver = random.choice(other_players)
        
        # Pick asset to offer
        proposer_assets = proposer.show_assets()
        if not proposer_assets:
            return -0.5
            
        asset1 = random.choice(proposer_assets)
        
        # Pick asset to request (or None)
        receiver_assets = receiver.show_assets()
        asset2 = random.choice(receiver_assets) if receiver_assets else None
        
        # Determine payment amounts (simplified)
        pay_amount = 0
        ask_amount = int(asset1.market_price * 0.8) if asset2 is None else 0
        
        # Store pending trade
        self.pending_trade = (proposer, receiver, asset1, asset2, pay_amount, ask_amount)
        print(f"Trade proposed: {proposer.player_name} offers {asset1.property_name} to {receiver.player_name}")
        
        return 0.1  # Small reward for initiating trade

    # ---------------------------
    # Heuristic players behavior
    # ---------------------------
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
                    proposal = self.trading_floor.proposition(asset1, asset2, pay_amount, ask_amount)
                    accept = player.see_proposition(proposal)
                else:
                    # Use heuristic
                    accept = self._heuristic_trade_decision(receiver, asset1, asset2, pay_amount, ask_amount)
                
                if accept:
                    self.trading_floor.open_trade(proposer, receiver, asset1, asset2, pay_amount, ask_amount)
                self.pending_trade = None
        
        # Handle active auction
        if self.active_auction is not None and player in self.active_auction.players:
            # Use player's make_bid if available
            if hasattr(player, 'make_bid') and callable(player.make_bid):
                # Player will be asked for bid in auction's start_auction method
                pass
            else:
                # Use heuristic bidding
                highest_bid = 0
                bid = self._heuristic_auction_bid(player, self.active_auction.property, highest_bid)
                if bid is None:
                    self.active_auction.players.remove(player)
        
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
            flag = "🤖 AGENT" if i == self.agent_index else "🎮 PLAYER"
            jail_status = "🔒 IN JAIL" if p.in_jail else ""
            print(f"{p.player_name:12s} {flag:10s} | Cash: ${p.capital:7.0f} | Pos: {p.position:2d} | Assets: {len(p.assets):2d} {jail_status}")
        
        print("-"*60)
        
        # Agent details
        if agent:
            net_worth = agent.capital + getattr(agent, 'net_asset_value', 0)
            print(f"Agent Net Worth: ${net_worth:8.0f}")
            
            if len(agent.assets) > 0:
                print("\nAgent Assets:")
                for asset in agent.assets:
                    asset_type = "🏠" if isinstance(asset, Property) else "🚂" if isinstance(asset, Railway) else "⚡"
                    mortgaged = " [MORTGAGED]" if getattr(asset, "is_mortgaged", False) else ""
                    houses = f" ({asset.houses}H)" if isinstance(asset, Property) and asset.houses > 0 else ""
                    hotel = " (HOTEL)" if isinstance(asset, Property) and asset.hotel else ""
                    print(f"  {asset_type} {asset.property_name:30s} ${asset.market_price:4.0f}{mortgaged}{houses}{hotel}")
        
        # Game state
        print("-"*60)
        print(f"Turn: {self.turn_count}/{self.max_turns}")
        if self.use_timer and self.timer:
            print(f"Time Left: {self.timer.time_left}s")
        if self.pending_trade:
            print("⚠️ Trade Pending")
        if self.active_auction:
            print(f"⚠️ Auction Active: {self.active_auction.property.property_name}")
        
        print("="*60 + "\n")

    def close(self):
        """Clean up resources including timer."""
        if self.timer:
            self.timer.stop()


# ===============================
# Helper function for external use
# ===============================
def create_monopoly_env(agent_index: int = 0, 
                        num_players: int = 4, 
                        use_timer: bool = False,
                        max_turns: int = 500,
                        seed: Optional[int] = None) -> MonopolyEnv:
    """
    Factory function to create a MonopolyEnv with common configurations.
    
    Parameters
    ----------
    agent_index : int
        Index of the RL agent (0 to num_players-1)
    num_players : int
        Total number of players (2-6 recommended)
    use_timer : bool
        Enable 90-second turn timer
    max_turns : int
        Maximum turns before episode truncation
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    MonopolyEnv
        Configured environment ready for training
    """
    return MonopolyEnv(
        agent_index=agent_index,
        num_players=num_players,
        seed=seed,
        max_turns=max_turns,
        use_timer=use_timer
    )


# ===============================
# Usage Documentation
# ===============================
"""
COMPLETE MONOPOLY RL ENVIRONMENT
================================

This environment integrates:
✓ Auction class from auction_floor.py
✓ Trading_floor class from trading_floor.py
✓ Timer90 class from turn_timer.py
✓ Full action masking (16 actions)
✓ Net worth-based reward shaping
✓ Proper game mechanics

TIMER SYSTEM:
- Set use_timer=True to enable 90-second turn limit
- Timer starts automatically when agent's turn begins
- If timer expires (reaches 0), agent receives -2.0 penalty
- Time remaining available in info["time_left"]

AUCTION SYSTEM (Using Auction class):
- Action.AUCTION starts an auction using Auction(property, players)
- All players participate via their make_bid() method
- Auction.start_auction(bank) handles the full auction process
- Heuristic players use _heuristic_auction_bid() as fallback

TRADING SYSTEM (Using Trading_floor class):
- Action.TRADE proposes trade using Trading_floor.proposition()
- Receiver evaluates via see_proposition() method
- Action.ACCEPT_TRADE executes via Trading_floor.open_trade()
- Heuristic players use _heuristic_trade_decision() as fallback

PLAYER METHOD INTEGRATION:
The environment automatically uses these Player methods if implemented:
- player.make_bid(property, highest_bid) -> int|None
- player.see_proposition(proposal) -> bool

If not implemented, fallback heuristics are used.

QUICK START:
    from MonopolyEnv import create_monopoly_env
    
    # Training configuration
    env = create_monopoly_env(agent_index=0, num_players=4, use_timer=False)
    
    # Training loop
    obs, info = env.reset()
    for episode in range(1000):
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            agent.learn(obs, action, reward, done)
        
        obs, info = env.reset()
    
    env.close()

OBSERVATION SPACE:
- Vector: [board_vec | player_vec | turn_info | action_mask]
- Total dimensions: ~board_specific + 164 + 4 + 16

ACTION SPACE:
- Discrete(16): BUY, TRADE, ACCEPT/DECLINE, AUCTION, BID_10/50/100,
  BUILD_HOUSE/HOTEL, SELL_HOUSE/HOTEL, MORTGAGE/UNMORTGAGE, JAIL actions

REWARD STRUCTURE:
- BUY_ASSET: +1
- BUILD_HOUSE: +3, BUILD_HOTEL: +5
- SELL_HOUSE: -3, SELL_HOTEL: -5
- FULL_SET: +5
- MORTGAGE: -1, UNMORTGAGE: +1
- Net worth delta: continuous feedback
- Invalid action: -1.0
- Timer expired: -2.0
- BANKRUPTCY: -500
- WIN_GAME: +1000

TRAINING TIPS:
1. Start without timer (use_timer=False) for faster learning
2. Use action masking in your RL algorithm
3. Monitor net worth as key performance indicator
4. Enable timer during evaluation
5. Consider curriculum learning (start with 2 players, increase gradually)

EXAMPLE WITH PPO:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Create environment
    def make_env():
        return create_monopoly_env(agent_index=0, num_players=4, use_timer=False)
    
    env = DummyVecEnv([make_env])
    
    # Train PPO agent
    model = PPO("MlpPolicy", env, verbose=1, 
                policy_kwargs={"net_arch": [256, 256]})
    model.learn(total_timesteps=1_000_000)
    
    # Evaluate
    eval_env = create_monopoly_env(agent_index=0, num_players=4, use_timer=True)
    obs, info = eval_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        eval_env.render()
        if done:
            break
    
    eval_env.close()
"""