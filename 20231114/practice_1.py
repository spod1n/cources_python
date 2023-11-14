'''> 7. Створіть клас Book для представлення книги.
Клас повинен мати атрибути title, author та year та методи для виведення інформації про книгу.'''

class Book:
    def __init__(self, title, author, year):
        self.title = title
        self.author = author
        self.year = year

    def info(self):
        print(f"Book '{self.title}'. {self.year} year by {self.author}")


book1 = Book('Duma Island', 'Stephen King', 2010)
book1.info()


'''> 8. Створіть клас RectangleArea для представлення площі прямокутника.
Клас повинен мати атрибути length та width та методи для обчислення площі прямокутника.'''
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height


figure1 = Rectangle(10, 20)
print(figure1.area())


'''> 9. Створіть клас Bank для представлення банку.
Клас повинен мати атрибути name та accounts та методи для відкриття нового рахунку,
закриття рахунку та виведення списку всіх рахунків.'''
class Bank:
    def __init__(self, name, accounts: list = []):
        self.name = name
        self.accounts = accounts

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

    def print_accounts(self):
        print(f'{self.name} was accounts:')
        for indx_acc, acc in enumerate(self.accounts):
            print(f"{indx_acc}: {indx_acc}")


acc1 = Bank('Klymentii')

acc1.create_account('0001')
acc1.create_account('0002')



''''> 10. Створіть клас Shape для представлення геометричної фігури.
Клас повинен мати атрибути name та sides та методи для виведення інформації про фігуру та обчислення периметру.
Класи Rectangle та Circle повинні бути нащадками класу Shape.'''