import random


class Card:
    """
    Represents a single Chance or Community Chest card.

    Attributes:
        description (str): Text shown to the player when the card is drawn.
        action (str): The action keyword that determines the card's effect.
            Supported actions include:
                - "move": Move the player to a specific board tile.
                - "move_back": Move the player backwards.
                - "jail": Send the player to jail.
                - "pay_bank": Player pays money to the bank.
                - "gain_bank": Player receives money from the bank.
                - "jail_free": Player receives a Get Out of Jail Free card.
                - "repair": Player pays based on number of houses/hotels owned.
                - "nearest_utility": Move to the next utility tile.
                - "nearest_railway": Move to the next railway tile.
        amount (int, optional): Numerical value associated with the card's action.
            Examples:
                - Tile index for movement cards.
                - Monetary value for gain/pay cards.
        target (dict, optional): Additional structured data for special actions.
            Example:
                For "repair" cards:
                    {"house": 40, "hotel": 115}
    """

    def __init__(self, description, action, amount=None, target=None):
        """
        Initialize a card object.

        Args:
            description (str): Text describing the card.
            action (str): Keyword defining the effect of the card.
            amount (int, optional): Numeric value used by the action.
            target (dict, optional): Additional parameters for the card effect.
        """
        self.description = description
        self.action = action
        self.amount = amount
        self.target = target


class ChanceDeck:
    """
    Represents the deck of Chance cards.

    This class handles:
        - Deck creation
        - Shuffling
        - Drawing cards
        - Automatic reshuffling when the deck is empty

    Attributes:
        cards (list[Card]): The full unshuffled set of Chance cards.
        deck (list[Card]): The current shuffled deck from which draws occur.
    """

    def __init__(self):
        """Initialize the Chance deck and shuffle it."""
        self.cards = [
            Card("Advance to GO.", action="move", amount=0),
            Card("Go to Jail. Do not pass GO.", action="jail"),
            Card("Go back 3 spaces.", action="move_back", amount=3),
            Card("Advance to Cape Town Station.", action="move", amount=5),
            Card("Advance to Bantry Bay.", action="move", amount=39),
            Card("Pay a R150 fine.", action="pay_bank", amount=150),
            Card("Your building loan matures. Receive R200.", action="gain_bank", amount=200),
            Card("Speeding fine. Pay R100.", action="pay_bank", amount=100),
            Card("Get out of jail free card.", action="jail_free"),
            Card("Take a trip to Pretoria Station.", action="move", amount=35),
            Card("Advance to the nearest Utility.", action="nearest_utility"),
            Card("Advance to the nearest Railway.", action="nearest_railway"),
        ]

        self.shuffle()

    def shuffle(self):
        """
        Shuffle the deck.

        Copies the original list of cards, shuffles it,
        and stores it in `self.deck`.
        """
        self.deck = self.cards.copy()
        random.shuffle(self.deck)

    def draw(self):
        """
        Draw a card from the deck.

        Automatically reshuffles if the deck becomes empty.

        Returns:
            Card: The drawn card.
        """
        if len(self.deck) == 0:
            self.shuffle()
        return self.deck.pop(0)


class CommunityChestDeck:
    """
    Represents the deck of Community Chest cards.

    Handles:
        - Deck creation
        - Shuffling
        - Card drawing
        - Auto-reshuffle when empty

    Attributes:
        cards (list[Card]): All Community Chest cards (unshuffled).
        deck (list[Card]): The active shuffled deck.
    """

    def __init__(self):
        """Initialize the Community Chest deck and shuffle it."""
        self.cards = [
            Card("Advance to GO.", action="move", amount=0),
            Card("Go to Jail. Do not pass GO.", action="jail"),
            Card("Bank error in your favor. Collect R200.", action="gain_bank", amount=200),
            Card("Doctor's fees. Pay R50.", action="pay_bank", amount=50),
            Card("Receive R25 consultancy fee.", action="gain_bank", amount=25),
            Card("You inherit R100.", action="gain_bank", amount=100),
            Card("From sale of stock, you receive R45.", action="gain_bank", amount=45),
            Card("Pay hospital fees of R100.", action="pay_bank", amount=100),
            Card("Life insurance matures. Receive R100.", action="gain_bank", amount=100),
            Card("Get out of jail free card.", action="jail_free"),
            Card(
                "You are assessed for street repairs. Pay R40 per house, R115 per hotel.",
                action="repair",
                target={"house": 40, "hotel": 115}
            ),
            Card("Christmas fund matures. Receive R100.", action="gain_bank", amount=100),
            Card("Pay school fees of R50.", action="pay_bank", amount=50),
        ]

        self.shuffle()

    def shuffle(self):
        """
        Shuffle the Community Chest deck.

        Creates a new randomized order of cards.
        """
        self.deck = self.cards.copy()
        random.shuffle(self.deck)

    def draw(self):
        """
        Draw a card from the deck.

        Reshuffles automatically when empty.

        Returns:
            Card: The card drawn from the top of the deck.
        """
        if len(self.deck) == 0:
            self.shuffle()
        return self.deck.pop(0)
