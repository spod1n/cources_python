class BankAccount(object):

    def __init__(self, account_number: str, balance: float = 0.0):

        self.account_number = account_number

        self.balance = balance

    def __str__(self):

        print(f'Bank_name: {self.account_number} Balance: {self.balance}')

    def add_money(self, amount):

        self.balance += amount

    def minus_money(self, amount):

        if amount > self.balance:

            print(f'No money: {self.balance - amount}')

        else:

            self.balance -= amount

