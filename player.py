class Player:
    """
    Represents a Monopoly player, including their position, assets, capital,
    and state-related variables such as jail status.

    Attributes
    ----------
    player_name : str
        The name of the player.
    capital : int
        The current liquid money the player has.
    assets : list
        A list of Property, Railway, or Utility objects owned by the player.
    net_asset_value : int
        The total asset value (liquid + mortgaged value + buildings).
    position : int
        The player's current board index (0–39).
    in_jail : bool
        Whether the player is currently in jail.
    jail_counter : int
        Number of turns spent in jail (max 3).
    has_jail_free_card : bool
        Whether the player holds a "Get Out of Jail Free" card.
    """

    def __init__(self, player_name, starting_capital):
        """
        Initialize a new player.

        Parameters
        ----------
        player_name : str
            The player's name.
        starting_capital : int
            The money the player starts with.
        """
        self.player_name = player_name
        self.capital = starting_capital
        self.assets = []
        self.net_asset_value = 0
        self.position = 0
        self.in_jail = False
        self.jail_counter = 0
        self.has_jail_free_card = False

    # ----------------------------------------------------------------------
    # FINANCIAL OPERATIONS
    # ----------------------------------------------------------------------

    def buy(self, asset, amount):
        """
        Purchase an asset and deduct cost from player's capital.

        Parameters
        ----------
        asset : Property or Railway or Utility
            The asset being purchased.
        amount : int
            The amount paid.

        Notes
        -----
        - Player gains mortgage value toward net_asset_value.
        """
        if self.capital < amount:
            print("Insufficient funds to buy")
            return

        self.capital -= int(amount)
        self.net_asset_value += asset.mortgage_value
        self.assets.append(asset)

    def build(self, asset, amount):
        """
        Build a house or hotel on a property.

        Parameters
        ----------
        asset : Property
            The property being upgraded.
        amount : int
            The cost of building.

        Notes
        -----
        - Net asset value increases by half the building cost.
        """
        if self.capital < amount:
            print("Insufficient funds to build")
            return

        self.capital -= int(amount)
        self.net_asset_value += int(amount / 2)

    def sell(self, amount):
        """
        Sell a house, hotel, or downgrade asset.

        Parameters
        ----------
        amount : int
            Sale revenue.

        Notes
        -----
        - Net asset value decreases by half the sold value.
        """
        self.capital += int(amount)
        self.net_asset_value -= int(amount / 2)

    def mortgage(self, amount):
        """
        Receive mortgage value for an asset.

        Parameters
        ----------
        amount : int
            Mortgage payout.
        """
        self.capital += int(amount)

    def unmortgage(self, amount):
        """
        Pay to unmortgage a property.

        Parameters
        ----------
        amount : int
            Unmortgage fee.

        Notes
        -----
        - If player lacks funds, unmortgage fails.
        """
        if self.capital < amount:
            print("Insufficient funds to unmortgage")
            return
        self.capital -= int(amount)

    def pay(self, amount):
        """
        Pay money (rent, tax, etc.).

        Parameters
        ----------
        amount : int
            Deducted from capital.
        """
        self.capital -= int(amount)

    def receive(self, amount):
        """
        Receive money (salary, rent, card reward, etc.).

        Parameters
        ----------
        amount : int
            Added to capital.
        """
        self.capital += int(amount)

    # ----------------------------------------------------------------------
    # STATUS CHECKS
    # ----------------------------------------------------------------------

    def show_assets(self):
        """Return list of owned assets."""
        return self.assets

    def see_proposition(self, prop):
        """
        Decide whether to accept a trade proposition.

        This method evaluates a trade proposal and returns whether the player
        accepts or rejects it based on strategic considerations.

        Parameters
        ----------
        prop : tuple(dict, dict)
            Proposed trade packet from Trading_floor.
            Format: (player1_offer, player2_offer)
            Each dict contains:
            - asset_name: str or None
            - asset_type: "Property", "Railway", "Utility", or None
            - owner: Player or None
            - mortgaged: bool
            - value: int (market price)
            - rent: int or str
            - extra: dict with additional info
            - asking_amount: int (money being offered/requested)

        Returns
        -------
        bool
            True if trade accepted, False otherwise.

        Strategy
        --------
        The default strategy evaluates:
        1. Net value comparison (what we get vs what we give)
        2. Capital constraints (can we afford it?)
        3. Strategic value (completing color sets, getting monopolies)
        4. Risk assessment (avoid becoming cash-poor)
        
        Notes
        -----
        - This is a heuristic implementation
        - RL agents should override this with learned behavior
        - Human players can override this for interactive play
        """
        if prop is None:
            return False
        
        # Unpack the trade proposition
        their_offer, our_offer = prop
        
        # Calculate what we're receiving
        receiving_asset_value = their_offer.get("value", 0) if their_offer.get("asset_name") else 0
        receiving_money = their_offer.get("asking_amount", 0)
        receiving_total = receiving_asset_value + receiving_money
        
        # Calculate what we're giving
        giving_asset_value = our_offer.get("value", 0) if our_offer.get("asset_name") else 0
        giving_money = our_offer.get("asking_amount", 0)
        giving_total = giving_asset_value + giving_money
        
        # Rule 1: Never accept if we can't afford the money we need to pay
        if giving_money > self.capital:
            return False
        
        # Rule 2: Don't accept if it leaves us with less than 20% of starting capital
        if self.capital - giving_money < 300:  # assuming 1500 starting capital
            return False
        
        # Rule 3: Check if the trade is worth it (need at least 10% better value)
        if receiving_total < giving_total * 0.9:
            return False
        
        # Rule 4: Prefer getting assets over giving them away (asset accumulation)
        if their_offer.get("asset_name") and not our_offer.get("asset_name"):
            # We're getting an asset without giving one - favorable
            if receiving_total >= giving_total * 0.7:
                return True
        
        # Rule 5: Check for strategic value (color set completion)
        if their_offer.get("asset_type") == "Property":
            # Check if this property helps complete a color set
            their_color = their_offer.get("extra", {}).get("set")
            if their_color:
                # Count how many we already own in that color
                our_properties_in_color = sum(
                    1 for asset in self.assets 
                    if hasattr(asset, 'property_set') and asset.property_set == their_color
                )
                
                # If this completes or nearly completes a set, more likely to accept
                if our_properties_in_color >= 1:  # We already own at least one in this set
                    if receiving_total >= giving_total * 0.8:
                        return True
        
        # Rule 6: Be more willing to trade if we have many assets (diversification)
        if len(self.assets) > 5 and receiving_total >= giving_total * 0.85:
            return True
        
        # Rule 7: Default - accept if getting significantly better value
        if receiving_total >= giving_total * 1.15:
            return True
        
        # Default: reject
        return False

    def make_bid(self, property, highest_bid):
        """
        Decide how much to bid in an auction.

        This method determines the player's bidding strategy during property auctions.

        Parameters
        ----------
        property : Property
            Asset being auctioned (Property, Railway, or Utility).
        highest_bid : int
            Current highest bid in the auction.

        Returns
        -------
        int or None
            Return a new bid amount (must be > highest_bid), or None to withdraw.

        Strategy
        --------
        The default strategy considers:
        1. Property value (don't overbid)
        2. Available capital (maintain liquidity)
        3. Strategic importance (railways, utilities, color sets)
        4. Current portfolio (diversification vs monopoly building)
        
        Notes
        -----
        - This is a heuristic implementation
        - RL agents should override this with learned behavior
        - Returns None to drop out of the auction
        """
        if property is None:
            return None
        
        # Get property details
        market_price = getattr(property, 'market_price', 0)
        property_type = type(property).__name__
        
        # Calculate maximum bid we're willing to make
        # Base: willing to pay up to 80% of market price
        base_max_bid = int(market_price * 0.8)
        
        # Adjust based on capital constraints
        # Never bid more than 30% of current capital to maintain liquidity
        capital_constraint = int(self.capital * 0.3)
        
        # Strategic adjustments
        strategic_multiplier = 1.0
        
        # 1. Railways are valuable (collect all 4)
        if property_type == "Railway":
            owned_railways = sum(1 for a in self.assets if type(a).__name__ == "Railway")
            if owned_railways > 0:
                # More railways we own, more valuable the next one
                strategic_multiplier = int(1.0 + (owned_railways * 0.15))
        
        # 2. Utilities are somewhat valuable (2 total)
        elif property_type == "Utility":
            owned_utilities = sum(1 for a in self.assets if type(a).__name__ == "Utility")
            if owned_utilities == 1:
                # Owning both utilities doubles the rent
                strategic_multiplier = 1.2
        
        # 3. Properties - check for color set potential
        elif property_type == "Property":
            property_set = getattr(property, 'property_set', None)
            if property_set:
                # Count how many we own in this color
                owned_in_set = sum(
                    1 for a in self.assets 
                    if hasattr(a, 'property_set') and a.property_set == property_set
                )
                
                if owned_in_set > 0:
                    # We own at least one in this set - more valuable
                    strategic_multiplier = int(1.0 + (owned_in_set * 0.2))
        
        # Calculate final max bid
        max_bid = min(base_max_bid * strategic_multiplier, capital_constraint)
        
        # If current highest bid already exceeds our max, drop out
        if highest_bid >= max_bid:
            return None
        
        # Determine our bid increment strategy
        remaining_room = max_bid - highest_bid
        
        if remaining_room < 10:
            return None  # Not enough room to bid
        
        # Bid in increments based on property value
        if market_price < 100:
            increment = 10
        elif market_price < 200:
            increment = 20
        elif market_price < 300:
            increment = 50
        else:
            increment = 100
        
        # Make our bid
        our_bid = highest_bid + increment
        
        # Don't exceed our maximum
        if our_bid > max_bid:
            our_bid = int(max_bid)
        
        # Ensure we can afford it
        if our_bid > self.capital:
            return None
        
        # Make sure it's actually higher than current bid
        if our_bid <= highest_bid:
            return None
        
        return our_bid

    def is_bankrupt(self):
        """
        Determine if the player has negative capital.

        Returns
        -------
        bool
            True if player has less than 0 capital.
        """
        return self.capital < 0

    def is_in_jail(self):
        """
        Update and check jail status.

        Returns
        -------
        bool
            True if player remains in jail.

        Notes
        -----
        - After 3 turns, the player is released automatically.
        """
        if self.in_jail:
            self.jail_counter += 1

        if self.jail_counter == 3:
            self.in_jail = False
            self.jail_counter = 0
            print(f"{self.player_name} is out of jail")

        return self.in_jail

    # ----------------------------------------------------------------------
    # RL ENVIRONMENT SUPPORT
    # ----------------------------------------------------------------------

    def player_observation(self):
        """
        Generate a structured observation of the player's current state.

        Returns
        -------
        dict
            A dictionary containing player's capital, net asset value,
            position, jail status, and owned assets.
        """
        obs = {
            "player_name": self.player_name,
            "capital": self.capital,
            "net_asset_value": self.net_asset_value,
            "position": self.position,
            "in_jail": self.in_jail,
            "owned_assets": self.assets,
        }
        
        return obs