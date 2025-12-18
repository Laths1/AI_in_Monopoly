class MESSAGES:
    # Movement and position messages
    MOVE_TO_POSITION = "{player_name} moves to position {amount}."
    MOVE_BACK = "{player_name} moves back {amount} spaces to position {position}."
    MOVES_TO_NEAREST = "{player_name} moves to nearest {tile_type} at position {new_pos}."
    
    # Jail messages
    GOES_TO_JAIL = "{player_name} goes to Jail!"
    USES_JAIL_CARD = "{player_name} uses a Jail Free card."
    RECEIVES_JAIL_CARD = "{player_name} receives a Jail Free card."
    PAYS_TO_LEAVE_JAIL = "{player_name} pays ${amount} to get out of jail."
    
    # Financial messages
    PAYS_TO_BANK = "{player_name} pays ${amount} to the bank."
    RECEIVES_FROM_BANK = "{player_name} receives ${amount} from the bank."
    PAYS_RENT = "{player_name} pays ${rent} to {owner_name}."
    PAYS_TAX = "{player_name} pays ${amount} in {tax_type} Tax."
    
    # Property messages
    BUYS_PROPERTY = "{player_name} buys {property_name} for ${price}."
    SELLS_PROPERTY = "{player_name} sells {property_name} for ${price}."
    MORTGAGES_PROPERTY = "{player_name} mortgages {property_name} for ${amount}."
    UNMORTGAGES_PROPERTY = "{player_name} unmortgages {property_name} for ${amount}."
    
    # Building messages
    BUILDS_HOUSE = "{player_name} builds a house on {property_name} for ${price}."
    BUILDS_HOTEL = "{player_name} builds a hotel on {property_name} for ${price}."
    SELLS_HOUSE = "{player_name} sells a house on {property_name} for ${price}."
    SELLS_HOTEL = "{player_name} sells a hotel on {property_name} for ${price}."
    REPAIRS_PROPERTIES = "{player_name} repairs properties for ${total}."
    
    # Card messages
    DRAWS_CHANCE = "{player_name} draws Chance card: {description}"
    DRAWS_COMMUNITY = "{player_name} draws Community Chest card: {description}"
    
    # Auction messages
    WINS_AUCTION = "{player_name} won auction for {property_name}"
    AUCTION_STARTED = "Auction started for {property_name}"
    
    # Trade messages
    TRADE_SUCCESSFUL = "Trade successful: {player1} ↔ {player2}"
    TRADE_FAILED = "Trade failed: {error}"
    TRADE_PROPOSED = "{proposer} proposes trade to {receiver}"
    
    # Game state messages
    PLAYER_INSOLVENT = "{player_name} is insolvent (${capital:.0f}). Attempting to restore solvency..."
    SOLD_HOTEL = "Sold hotel on {property_name} for ${sell_price:.0f}"
    SOLD_HOUSE = " Sold 1 house on {property_name} for ${sell_price:.0f}"
    MORTGAGED_ASSET = "Mortgaged {property_name} for ${mortgage_value:.0f}"
    NO_MORE_ASSETS = "No more assets to liquidate"
    MAX_ITERATIONS_WARNING = "Warning: Max liquidation iterations reached!"
    PLAYER_BANKRUPT = "{player_name} has declared BANKRUPTCY! (${capital:.0f})"
    PLAYER_RESTORED = "{player_name} restored solvency! New balance: ${capital:.0f}\n"
    
    # Timer messages
    TIMER_TIMEOUT = "{player_name}'s turn timed out!"
    
    # Rendering messages
    GAME_STATE_HEADER = "MONOPOLY GAME STATE"
    PLAYER_INFO = "{player_name:12s} {flag:10s} | Cash: ${capital:7.0f} | Pos: {position:2d} | Assets: {assets:2d} {jail_status}"
    AGENT_NET_WORTH = "Agent Net Worth: ${net_worth:8.0f}"
    AGENT_ASSETS_HEADER = "Agent Assets:"
    ASSET_DETAILS = "  {asset_type} {property_name:30s} ${market_price:4.0f}{mortgaged}{houses}{hotel}"
    TURN_INFO = "Turn: {turn_count}/{max_turns}"
    TIME_LEFT = "Time Left: {time_left}s"
    TRADE_PENDING = "Trade Pending"
    AUCTION_ACTIVE = "Auction Active: {property_name}"