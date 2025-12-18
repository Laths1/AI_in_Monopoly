class Auction:
    """
    Handles property auctions in the Monopoly game.

    Auctions occur when a player declines to buy an unowned property.
    All players have a chance to bid. Players may drop out at any time.

    The auction ends when:
    - Only one player remains, or
    - No new bids are made

    Attributes
    ----------
    property : Property
        The property being auctioned.
    players : list[Player]
        List of players participating in the auction.
    """

    def __init__(self, property, players):
        """
        Initialize the auction with a target property and eligible players.

        Parameters
        ----------
        property : Property
            The property being auctioned.
        players : list[Player]
            The players allowed to place bids.
        """
        self.property = property
        self.players = players[:]  # copy to avoid external mutation

    def start_auction(self, bank):
        """
        Run the auction until a winner emerges or all players withdraw.

        Parameters
        ----------
        bank : Bank
            The bank object used to transfer money when the auction is completed.

        Returns
        -------
        Player or None
            The winning bidder, or None if no one bid.

        Notes
        -----
        - Each player decides their own bid via player.make_bid(property, current_highest).
        - A player returns `None` to withdraw from the auction.
        - Highest bid wins once only one bidder is left.
        """
        print(f"Auction started for {self.property.property_name}")

        highest_bid = 0
        highest_bidder = None

        # Continue as long as more than one player is still bidding
        while len(self.players) > 1:
            # Track players who withdraw
            withdrawing_players = []

            for player in self.players:
                bid = player.make_bid(self.property, highest_bid)

                # Player leaves auction
                if bid is None:
                    print(f"{player.player_name} pulled out.")
                    withdrawing_players.append(player)
                    continue

                # New highest bid
                if bid > highest_bid:
                    highest_bid = bid
                    highest_bidder = player

            # Remove withdrawn players safely
            for p in withdrawing_players:
                self.players.remove(p)

            # If nobody raised the bid this round, end early
            if highest_bidder is None:
                break

        # Finalize results
        if highest_bidder:
            self.property.buy(highest_bidder)
            highest_bidder.buy(self.property, highest_bid)
            bank.buy(highest_bid)

            print(
                f"{highest_bidder.player_name} won the auction for "
                f"{self.property.property_name} with a bid of {highest_bid}"
            )

            return highest_bidder

        print("No bids were made in the auction.")
        return None
