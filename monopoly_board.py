"""
Monopoly Board Module
---------------------
This module defines all classes representing the Monopoly board, including:

- Property
- Railway
- Utility
- Square
- Board

This version includes complete internal documentation and docstrings.
"""

from rewards import *

class Property:
    """
    Represents a Monopoly property that can be bought, mortgaged,
    upgraded with houses and hotels, and that generates rent.

    Attributes
    ----------
    property_name : str
        Name of the property.
    market_price : int
        Cost to buy this property from the bank.
    house_price : int
        Cost of building a house on this property.
    hotel_price : int
        Cost of building a hotel (2× house price).
    property_set : str
        The color group this property belongs to.
    mortgage_value : float
        Amount received when mortgaging this property (half the market price).
    rent : float
        Current rent owed when another player lands here.
    house/hotel : int/bool
        Current building status (0–4 houses, or a hotel).
    sold : bool
        True once the property is purchased.
    is_mortgaged : bool
        If True, property generates no rent.
    owner : Player or None
        Current owner of this property.
    """

    def __init__(self, property_name, market_price, house_price, property_set):
        self.property_name = property_name
        self.market_price = market_price
        self.property_set = property_set
        self.mortgage_value = int(market_price / 2)
        self.rent = int(market_price / 10)
        self.house_price = house_price
        self.hotel_price = house_price * 2
        self.hotel = False
        self.sold = False
        self.has_full_set = False
        self.is_mortgaged = False
        self.houses = 0
        self.owner = None

    def mortgage_property(self, owner):
        """
        Mortgage this property if the owner matches.

        Returns
        -------
        int
            reward.
        str
            Error message if the owner does not match.
        """
        if self.owner == owner and self.sold:
            self.is_mortgaged = True
            return Reward.MORTGAGE_ASSET
        return Reward.NO_REWARD

    def unmortgage_property(self, owner):
        """
        Unmortgage this property if owned.

        Returns reward or error.
        """
        if self.owner == owner and self.sold:
            self.is_mortgaged = False
            return Reward.UNMORTGAGE_ASSET
        return Reward.NO_REWARD

    def is_full_set(self):
        """Return True if the property’s set is fully owned."""
        return self.has_full_set

    def trade(self, new_owner):
        """Transfer ownership during a trade."""
        self.owner = new_owner

    def buy(self, owner):
        """
        Mark the property as sold and assign ownership.

        Returns
        -------
        int : reward
        """
        self.sold = True
        self.owner = owner
        return Reward.BUY_ASSET

    def build_house(self, owner):
        """
        Build a house if:
        - Owner matches
        - Property is sold
        - Player has full set
        - Houses < 4

        Rent increases by ×2.5.
        Returns reward or error.

        """
        if self.owner == owner and self.sold and self.has_full_set and self.houses < 4:
            self.houses += 1
            self.rent = int(self.rent * 2.5)
            return Reward.BUILD_HOUSE
        return Reward.NO_REWARD

    def sell_house(self, owner):
        """
        Sell a house if one exists.

        Rent decreases accordingly.
        returns reward or error message
        """
        if self.owner == owner and self.sold and self.has_full_set and self.houses > 0:
            self.houses -= 1
            self.rent = int(self.rent / 2.5)
            return Reward.SELL_HOUSE
        return Reward.NO_REWARD

    def build_hotel(self, owner):
        """
        Build a hotel if:
        - Owner matches
        - Full set is owned
        - Exactly 4 houses present

        return reward or error message
        """
        if self.owner == owner and self.sold and self.has_full_set and self.houses == 4:
            self.hotel = True
            self.rent = int(self.rent * 1.5)
            return Reward.BUILD_HOTEL
        return Reward.NO_REWARD

    def sell_hotel(self, owner):
        """
        Sell the hotel and reduce the rent.

        returns reward or error message
        """
        if self.owner == owner and self.sold and self.has_full_set and self.hotel:
            self.hotel = False
            self.rent = int(self.rent / 1.5)
            return Reward.SELL_HOTEL
        return Reward.NO_REWARD

    def is_sold(self):
        """Return True if the property has been purchased."""
        return self.sold

    def pay_rent(self):
        """
        Calculate rent owed when another player lands here.

        Returns zero if mortgaged.
        """
        if self.sold:
            return 0 if self.is_mortgaged else self.rent
        return 0


class Railway(Property):
    """
    Represents a railway station.

    Uses the same structure as Property, but:
    - No houses or hotels.
    - Rent depends on number of railways owned (handled elsewhere).
    """

    def __init__(self, name, price):
        super().__init__(name, price, 0, "Railway")


class Utility(Property):
    """
    Represents an electricity or water utility.

    Differences:
    - No houses or hotels.
    - Rent depends on dice roll.
    """

    def __init__(self, name, price):
        super().__init__(name, price, 0, "Utility")

    def pay_rent(self, dice_roll):
        """
        Rent formula: dice_roll × 10  
        Returns 0 if mortgaged or unsold.
        """
        return (dice_roll * 10) if self.sold and not self.is_mortgaged else 0


class Square:
    """
    Represents any square on the Monopoly board.

    Parameters
    ----------
    position : int
        Board index 0–39.
    type : str
        "Property", "Railway", "Utility", or "Special".
    obj : object or None
        Associated object (Property/Railway/Utility/string).
    """

    def __init__(self, position, type, obj=None):
        self.position = position
        self.type = type
        self.obj = obj

    def get_type(self):
        """Return the square type."""
        return self.type

    def get_obj(self):
        """Return the associated object."""
        return self.obj

    def get_position(self):
        """Return the board index."""
        return self.position


class Board:
    """
    Represents the full 40-square Monopoly board.

    This class:
    - Constructs all properties, railways, and utilities.
    - Defines all special squares (GO, Chance, Jail, taxes).
    - Builds a dictionary mapping board positions to Square objects.
    """

    def __init__(self):
        # -----------------------------------
        # All Properties
        # -----------------------------------
        self.properties = {
            # Brown set
            1: Property("Adderley Street", 60, 50, "Brown"),
            3: Property("Buitengracht Street", 60, 50, "Brown"),

            # Light Blue
            6: Property("Durban Road", 100, 50, "Light Blue"),
            8: Property("West Street", 100, 50, "Light Blue"),
            9: Property("West Street North", 120, 50, "Light Blue"),

            # Pink
            11: Property("Commissioner Street", 140, 100, "Pink"),
            13: Property("Market Street", 140, 100, "Pink"),
            14: Property("Eloff Street", 160, 100, "Pink"),

            # Orange
            16: Property("Victoria Embankment", 180, 100, "Orange"),
            18: Property("West Street", 180, 100, "Orange"),
            19: Property("Smith Street", 200, 100, "Orange"),

            # Red
            21: Property("Voortrekker Road", 220, 150, "Red"),
            23: Property("Langa", 220, 150, "Red"),
            24: Property("Khayelitsha", 240, 150, "Red"),

            # Yellow
            26: Property("Maitland", 260, 150, "Yellow"),
            27: Property("Parow", 260, 150, "Yellow"),
            29: Property("Goodwood", 280, 150, "Yellow"),

            # Green
            31: Property("Rondebosch", 300, 200, "Green"),
            32: Property("Claremont", 300, 200, "Green"),
            34: Property("Newlands", 320, 200, "Green"),

            # Dark Blue
            37: Property("Camps Bay", 350, 200, "Dark Blue"),
            39: Property("Bantry Bay", 400, 200, "Dark Blue"),
        }

        # -----------------------------------
        # Railways
        # -----------------------------------
        self.railways = {
            5: Railway("Cape Town Station", 200),
            15: Railway("Johannesburg Park Station", 200),
            25: Railway("Durban Station", 200),
            35: Railway("Pretoria Station", 200),
        }

        # -----------------------------------
        # Utilities
        # -----------------------------------
        self.utilities = {
            12: Utility("Eskom Electricity", 150),
            28: Utility("Rand Water", 150),
        }

        # -----------------------------------
        # Special Squares
        # -----------------------------------
        self.special_squares = {
            0: "GO",
            2: "Community Chest",
            4: "Income Tax",
            7: "Chance",
            10: "Jail / Just Visiting",
            17: "Community Chest",
            20: "Free Parking",
            22: "Chance",
            30: "Go To Jail",
            33: "Community Chest",
            36: "Chance",
            38: "Luxury Tax",
        }

        # -----------------------------------
        # Build complete board dictionary
        # -----------------------------------
        self.board = {}

        for pos in range(40):
            if pos in self.properties:
                self.board[pos] = Square(pos, "Property", self.properties[pos])

            elif pos in self.railways:
                self.board[pos] = Square(pos, "Railway", self.railways[pos])

            elif pos in self.utilities:
                self.board[pos] = Square(pos, "Utility", self.utilities[pos])

            elif pos in self.special_squares:
                self.board[pos] = Square(pos, "Special", self.special_squares[pos])

            else:
                # Empty special squares (e.g., Free Parking equivalents)
                self.board[pos] = Square(pos, "Special", None)

    def see_board(self):
        """
        Print a readable representation of the Monopoly board.

        Shows:
        - Position index
        - Type of square
        - Name of property/railway/utility or special tile label
        """
        for pos in range(40):
            square = self.board[pos]

            if square.type in ["Property", "Railway", "Utility"]:
                print(f"{pos}: {square.type} → {square.obj.property_name}")

            elif square.obj:
                print(f"{pos}: {square.type} → {square.obj}")

            else:
                print(f"{pos}: {square.type}")

    # ----------------------------------------------------------------------
    # RL ENVIRONMENT SUPPORT
    # ----------------------------------------------------------------------

    def board_observation(self):
        """
        Return board:
        - unsold assets
        - sold assets
        - martgaged assets
        - asset prices
        """
        
        board_info = {
            "unsold_properties": [],
            "sold_properties": [],
            "mortgaged_properties": [],
            "property_prices": {},
            "unsold_railways": [],
            "sold_railways": [],
            "mortgaged_railways": [],
            "railway_prices": {},
            "unsold_utilities": [],
            "sold_utilities": [],
            "mortgaged_utilities": [],
            "utility_prices": {},
        }

        # Properties
        for prop in self.properties.values():
            if not prop.is_sold():
                board_info["unsold_properties"].append(prop.property_name)
            else:
                board_info["sold_properties"].append(prop.property_name)
                if prop.is_mortgaged:
                    board_info["mortgaged_properties"].append(prop.property_name)
            board_info["property_prices"][prop.property_name] = prop.market_price

        # Railways
        for rail in self.railways.values():
            if not rail.is_sold():
                board_info["unsold_railways"].append(rail.property_name)
            else:
                board_info["sold_railways"].append(rail.property_name)
                if rail.is_mortgaged:
                    board_info["mortgaged_railways"].append(rail.property_name)
            board_info["railway_prices"][rail.property_name] = rail.market_price

        # Utilities
        for util in self.utilities.values():
            if not util.is_sold():
                board_info["unsold_utilities"].append(util.property_name)
            else:
                board_info["sold_utilities"].append(util.property_name)
                if util.is_mortgaged:
                    board_info["mortgaged_utilities"].append(util.property_name)
            board_info["utility_prices"][util.property_name] = util.market_price

        return board_info