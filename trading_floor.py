from monopoly_board import Property, Railway, Utility

class Trading_floor:
    """
    Handles all trading interactions between players in the Monopoly game.

    The Trading_floor is responsible for:
    - Packaging assets into a structured trade dictionary
    - Creating bilateral trade proposals between players
    - Executing accepted trades between two players

    This class does NOT decide whether a trade is "good" — that decision is
    delegated to the player AI or human-controlled `see_proposition()` method.
    """

    def trade(self, asset, amount):
        """
        Convert an asset into a standardized dictionary for trading.

        Parameters
        ----------
        asset : Property | Railway | Utility | None
            The asset being traded. If None, returns an empty trade structure.
        amount : int
            The monetary value included with this part of the trade
            (the amount the offering player desires or offers).

        Returns
        -------
        dict
            A trade packet containing:
            - asset_name: str or None
            - asset_type: "Property", "Railway", "Utility", or None
            - owner: Player or None
            - mortgaged: bool
            - value: int (market price)
            - rent: int or str
            - extra: additional type-specific metadata
            - asking_amount: the offered or requested money
        """
        trade_info = {
            "asset_name": None,
            "asset_type": None,
            "owner": None,
            "mortgaged": None,
            "value": None,
            "rent": None,
            "extra": {},
            "asking_amount": amount
        }

        if asset is None:
            return trade_info

        # PROPERTIES -----------------------------------------------------------
        if isinstance(asset, Property):
            trade_info["asset_name"] = asset.property_name
            trade_info["asset_type"] = "Property"
            trade_info["owner"] = asset.owner
            trade_info["mortgaged"] = asset.is_mortgaged
            trade_info["value"] = asset.market_price
            trade_info["rent"] = asset.rent

            trade_info["extra"] = {
                "houses": asset.houses,
                "hotel": asset.hotel,
                "set": asset.property_set,
                "house_price": asset.house_price,
            }

        # RAILWAYS -------------------------------------------------------------
        elif isinstance(asset, Railway):
            trade_info["asset_name"] = asset.name
            trade_info["asset_type"] = "Railway"
            trade_info["owner"] = asset.owner
            trade_info["mortgaged"] = asset.is_mortgaged
            trade_info["value"] = asset.market_price

            if asset.owner:
                num_owned = sum(1 for r in asset.owner.railways if r.sold)
            else:
                num_owned = 0

            trade_info["rent"] = 25 * num_owned
            trade_info["extra"] = {"owned_railways": num_owned}

        # UTILITIES ------------------------------------------------------------
        elif isinstance(asset, Utility):
            trade_info["asset_name"] = asset.name
            trade_info["asset_type"] = "Utility"
            trade_info["owner"] = asset.owner
            trade_info["mortgaged"] = asset.is_mortgaged
            trade_info["value"] = asset.market_price

            if asset.owner:
                util_owned = sum(1 for u in asset.owner.utilities if u.sold)
            else:
                util_owned = 0

            multiplier = 4 if util_owned == 1 else 10
            trade_info["rent"] = f"{multiplier} × dice roll"

            trade_info["extra"] = {
                "owned_utilities": util_owned,
                "multiplier": multiplier,
            }

        return trade_info


    def proposition(self, asset1, asset2, pay_amount, asking_amount):
        """
        Create a full two-party trade proposal.

        Parameters
        ----------
        asset1 : Property | Railway | Utility | None
            Asset offered by player1.
        asset2 : Property | Railway | Utility | None
            Asset offered by player2.
        pay_amount : int
            Money paid by player1.
        asking_amount : int
            Money paid by player2.

        Returns
        -------
        tuple(dict, dict)
            Two trade packets: (player1_offer, player2_offer)
        """
        return (
            self.trade(asset1, pay_amount),
            self.trade(asset2, asking_amount)
        )


    def open_trade(self, player1, player2, asset1, asset2, pay_amount, asking_amount):
        """
        Conduct a trade between two players.

        Parameters
        ----------
        player1 : Player
            The player initiating the trade.
        player2 : Player
            The receiving player who decides to accept/reject.
        asset1 : Property | Railway | Utility | None
            Asset offered by player1.
        asset2 : Property | Railway | Utility | None
            Asset offered by player2.
        pay_amount : int
            Money given by player1.
        asking_amount : int
            Money given by player2.

        Returns
        -------
        str
            "Trade successful!" or "Trade rejected."

        Notes
        -----
        - The method calls `player2.see_proposition()` to determine acceptance.
        - If accepted:
            * Money is exchanged.
            * Asset ownership is swapped.
        - If rejected, nothing happens.
        """
        
        proposal = self.proposition(asset1, asset2, pay_amount, asking_amount)

        # Player 2 decides
        accepted = player2.see_proposition(proposal)

        if not accepted:
            return "Trade rejected."

        # Money exchange
        player1.pay(pay_amount)
        player1.receive(asking_amount)

        player2.pay(asking_amount)
        player2.receive(pay_amount)

        # Swap owners
        if asset1:
            asset1.trade(player2)
        if asset2:
            asset2.trade(player1)

        return "Trade successful!"
