import random
import numpy as np
from wandb import watch

from monopoly_board import *
from bank import *
from trading_floor import *
from player import *   
from auction_floor import * 
from cards import *
from rewards import *
from actions import *
from turn_timer import *

def roll_dice():
    """
    Roll a 6-sided dice.

    Returns:
        int: A random integer between 1 and 6.
    """
    return random.randint(1, 6)


def resolve_card(player, card, board, bank, players, dice_value):
    """
    Execute the effect of a Chance or Community Chest card.

    Args:
        player (Player): The current player drawing the card.
        card (Card): The card drawn, containing action + parameters.
        board (Board): Game board object.
        bank (Bank): Bank object managing money flow.
        players (list[Player]): List of all players.
        dice_value (int): Last dice roll, used for utilities.

    Card Actions Supported:
        - move: Move to a specific tile.
        - move_back: Move backwards.
        - jail: Send to jail (unless player has Get Out of Jail Free).
        - pay_bank: Pay the bank.
        - gain_bank: Receive money from the bank.
        - jail_free: Grant a Get Out of Jail Free card.
        - repair: Pay for house/hotel repairs.
        - nearest_railway: Move to nearest railway.
        - nearest_utility: Move to nearest utility.
    """
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
    """
    Resolve the action associated with the tile a player lands on.

    Args:
        current_player (Player): Player whose turn it is.
        tile (Tile): The board tile landed on.
        bank (Bank): Bank handling payments and purchases.
        board (Board): Game board containing tile information.
        players (list[Player]): All players.
        dice (int): Dice result.
        chance_deck (ChanceDeck): Chance deck instance.
        community_deck (CommunityChestDeck): Community Chest deck.
    """
    if tile.get_type() == "Property":
        prop = tile.get_obj()
        print(f"Landed on Property: {prop.property_name}")

        if not prop.sold:
            price = prop.market_price
            if current_player.capital >= price:
                current_player.buy(prop, price)
                bank.buy(price)
                prop.buy(current_player)
                print(f"{current_player.player_name} bought {prop.property_name} for {price}")
            else:
                print(f"{current_player.player_name} cannot afford {prop.property_name}, starting auction")
                # Auction could happen here
        else:
            if prop.owner != current_player:
                rent = prop.pay_rent()
                current_player.pay(rent)
                prop.owner.receive(rent)
                print(f"{current_player.player_name} paid rent: {rent}")

    elif tile.get_type() == "Railway":
        rail = tile.get_obj()
        print(f"Landed on Railway: {rail.property_name}")

        if not rail.sold:
            price = rail.market_price
            if current_player.capital >= price:
                current_player.buy(rail, price)
                bank.buy(price)
                rail.buy(current_player)
            else:
                print(f"{current_player.player_name} cannot afford {rail.property_name}, starting auction")
        else:
            if rail.owner != current_player:
                rent = rail.pay_rent()
                current_player.pay(rent)
                rail.owner.receive(rent)

    elif tile.get_type() == "Utility":
        util = tile.get_obj()
        print(f"Landed on Utility: {util.property_name}")

        if not util.sold:
            price = util.market_price
            if current_player.capital >= price:
                current_player.buy(util, price)
                bank.buy(price)
                util.buy(current_player)
            else:
                print(f"{current_player.player_name} cannot afford {util.property_name}, starting auction")
        else:
            if util.owner != current_player:
                rent = util.pay_rent(dice)
                current_player.pay(rent)
                util.owner.receive(rent)

    elif tile.get_type() == "Special":
        print(f"Special Tile: {tile.get_obj()}")

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
            print(f"Chance Card: {card.description}")
            resolve_card(current_player, card, board, bank, players, dice)

        elif tile.obj == "Community Chest":
            card = community_deck.draw()
            print(f"Community Chest: {card.description}")
            resolve_card(current_player, card, board, bank, players, dice)


def check_set_ownership(player):
    """
    Check if a player owns full color sets required to build houses/hotels.

    Args:
        player (Player): Player whose ownership is being evaluated.

    Returns:
        list[str]: Names of full color sets owned.
    """
    player_properties = [asset for asset in player.show_assets() if isinstance(asset, Property)]

    Brown_cnt, Light_Blue_cnt, Pink_cnt, Orange_cnt, Red_cnt, Yellow_cnt, Green_cnt, Dark_Blue_cnt = 0, 0, 0, 0, 0, 0, 0, 0 
    for prop in player_properties: 
        match prop.property_set: 
            case "Brown": Brown_cnt += 1 
            case "Light Blue": Light_Blue_cnt += 1 
            case "Pink": Pink_cnt += 1 
            case "Orange": Orange_cnt += 1 
            case "Red": Red_cnt += 1 
            case "Yellow": Yellow_cnt += 1 
            case "Green": Green_cnt += 1 
            case "Dark Blue": Dark_Blue_cnt += 1 
            
    full_sets = [] 
    
    if Brown_cnt == 2:
        full_sets.append("Brown") 
    if Light_Blue_cnt == 3:
        full_sets.append("Light Blue")
    if Pink_cnt == 3:
        full_sets.append("Pink") 
    if Orange_cnt == 3:
        full_sets.append("Orange") 
    if Red_cnt == 3:
        full_sets.append("Red")
    if Yellow_cnt == 3:
        full_sets.append("Yellow") 
    if Green_cnt == 3:
        full_sets.append("Green") 
    if Dark_Blue_cnt == 2:
        full_sets.append("Dark Blue")
        
    for prop in player_properties: 
        if prop.property_set in full_sets:
            prop.has_full_set = True 
            
    return full_sets

def build(current_player, bank):
    """
    Attempt to build houses or hotels on all eligible properties the player owns.

    Args:
        current_player (Player): The player taking the action.
        bank (Bank): Bank funding construction costs.
    """
    for asset in current_player.show_assets():
        if isinstance(asset, Property):

            # Build house
            if asset.has_full_set and not asset.is_mortgaged and asset.houses < 4 and not asset.hotel:
                cost = asset.house_price
                if current_player.capital / 2 >= cost:
                    current_player.build(asset, cost)
                    bank.buy(cost)
                    asset.build_house(current_player)

            # Build hotel
            if asset.has_full_set and not asset.is_mortgaged and asset.houses == 4 and not asset.hotel:
                cost = asset.hotel_price
                if current_player.capital / 2 >= cost:
                    current_player.build(asset, cost)
                    bank.buy(cost)
                    asset.build_hotel(current_player)


def check_player_solvency(current_player, bank, players):
    """
    Attempt to restore player's solvency by selling assets, mortgaging,
    or ultimately declaring bankruptcy.

    Args:
        current_player (Player): Player being evaluated.
        bank (Bank): Bank receiving mortgage/sale payments.
        players (list[Player]): Player list, used to remove bankrupt players.

    returns:
        int: -500 if player is bankrupt, 0 otherwise.
    """
    print("Check player's solvency")

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

                elif isinstance(asset, Railway) or isinstance(asset, Utility):
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

# ----------------------------------------------------------------------
# RL ENVIRONMENT SUPPORT
# ----------------------------------------------------------------------

def observation(current_player, board):
    dice = roll_dice()
    print(f"Dice rolled: {dice}")

    print(f"Capital: {current_player.capital}")
    print(f"Assets: {len(current_player.show_assets())}")

    old_pos = current_player.position
    new_pos = (old_pos + dice) % 40
    current_player.position = new_pos
    tile = board.board[new_pos]

    observation = { "dice_roll": dice,
                    "old_position": old_pos,
                    "new_position": new_pos,
                    "capital": current_player.capital,
                    "tile": tile,
                    }
    
    return observation

# ---------------------------
#      MAIN GAME LOOP
# ---------------------------

if __name__ == "__main__":
    """
    Main game execution loop that simulates an unending Monopoly
    game until one player remains or the bank collapses.
    """
    board = Board()
    bank = Bank(salary=200)
    trading = Trading_floor()
    chance_cards = ChanceDeck()
    community_cards = CommunityChestDeck()

    players = [
        Player("Player1", 1500),
        Player("Player2", 1500),
        Player("Player3", 1500),
        Player("Player4", 1500),
        Player("Player5", 1500),
        Player("Player6", 1500),
    ]

    i = random.randint(0, 5)

    while True:
        # Bank Collapse Condition
        if not bank.bank_solvency():
            print("Economy has collapsed")
            print(f"Players left: {len(players)}")
            break

        current_player = players[i]

        # check if player is in jail
        if current_player.is_in_jail():
            print(f"{current_player.player_name} is in jail")
            i = (i + 1) % len(players)
            continue

        print(f"\n===== {current_player.player_name}'s Turn =====")

        # state observations
        obs = observation(current_player, board)
        old_pos = obs["old_position"]
        new_pos = obs["new_position"]
        dice = obs["dice_roll"]
        tile = obs["tile"]

        # Passing GO
        if old_pos + dice >= 40:
            salary = bank.go_salary()
            current_player.receive(salary)
            print(f"{current_player.player_name} passed GO (+{salary})")

        print(f"Moved to tile {new_pos}")

        # player action in square
        tile_interaction(current_player, tile, bank, board, players, dice, chance_cards, community_cards)

        # building houses/hotels
        full_sets = check_set_ownership(current_player)
        if full_sets:
            build(current_player, bank)

        # check solvency
        check_player_solvency(current_player, bank, players)
                
        # Win Condition
        if len(players) == 1:
            print(f"{players[0].player_name} wins!")
            reward = Reward.WIN_GAME
            break

        # Next player's turn
        i = (i + 1) % len(players)
