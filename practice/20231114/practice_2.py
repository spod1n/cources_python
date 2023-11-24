'''> 9. Створіть клас Bank для представлення банку.
Клас повинен мати атрибути name та accounts та методи для відкриття нового рахунку,
закриття рахунку та виведення списку всіх рахунків.'''
class Bank:
    def __init__(self, name):
        self.name = name
        self.accounts = []

    def create_account(self, account):
        if account not in self.accounts:
            self.accounts.append(account)
            print(f"Account '{account}' was created.")
        else:
            print(f'Error! You already have such an account.')

    def delete_account(self, account):
        if account in self.accounts:
            self.accounts.remove(account)
            print(f"Account '{account}' was deleted.")
        else:
            print(f"Error! Account '{account}' bot found!")

    def print_accounts(self):
        print(f'{self.name} was accounts:')
        for indx_acc, acc in enumerate(self.accounts):
            print(f"{indx_acc}: {acc}")


acc1 = Bank('Klymentii')

acc1.create_account('0001')
acc1.create_account('0002')
acc1.delete_account('0002')

acc1.print_accounts()

acc2 = Bank('Olya')
acc2.create_account('0001')
acc2.print_accounts()
