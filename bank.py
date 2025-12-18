class Bank:
    """
    Represents the central bank in the Monopoly game.

    The Bank is responsible for:
    - Paying players when they pass GO
    - Receiving money from property purchases, taxes, and fees
    - Handling mortgages and unmortgages
    - Tracking its solvency over the course of the game

    Attributes
    ----------
    salary : int
        The amount paid to a player for passing GO.
    bank_capital : int
        Total money currently held by the bank.
    """

    def __init__(self, salary):
        """
        Initialize the Bank with a GO salary and a starting capital.

        Parameters
        ----------
        salary : int
            The amount players receive when passing GO.
        """
        self.salary = salary
        self.bank_capital = 20_000

    def go_salary(self):
        """
        Pays a player the GO salary and reduces the bank's capital.

        Returns
        -------
        int
            The salary paid to the player.
        """
        self.bank_capital -= self.salary
        return self.salary

    def mortgage(self, amount):
        """
        Processes mortgage payments to the bank.

        Parameters
        ----------
        amount : int
            Amount received by the bank from a mortgage.
        """
        self.bank_capital += int(amount)

    def unmortgage(self, amount):
        """
        Processes unmortgage fees paid from the bank.

        Parameters
        ----------
        amount : int
            Amount the bank pays out to release a mortgage.
        """
        self.bank_capital -= int(amount)

    def buy(self, amount):
        """
        Adds capital to the bank when a player buys a property.

        Parameters
        ----------
        amount : int
            The purchase price the bank receives.
        """
        self.bank_capital += int(amount)
    
    def sell(self, amount):
        """
        Reduces the bank's capital when paying a player (e.g., selling houses).

        Parameters
        ----------
        amount : int
            The amount paid out by the bank.
        """
        self.bank_capital -= int(amount)

    def bank_solvency(self):
        """
        Checks whether the bank is still solvent.

        Returns
        -------
        bool
            True if the bank has remaining capital, False otherwise.
        """
        return self.bank_capital > 0
