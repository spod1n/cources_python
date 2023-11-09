"""Task 1. Створіть клас Rectangle для представлення прямокутника. Клас повинен мати атрибути
width та height і методи для обчислення площі та периметру прямокутника."""

import math

class Rectangle:
    """
    Клас Rectangle представляє прямокутник з атрибутами width та height.

    :ivar width: Ширина прямокутника.
    :vartype width: float
    :ivar height: Висота прямокутника.
    :vartype height: float
    """

    def __init__(self, width, height):
        """
        Конструктор класу Rectangle.

        :param width: Ширина прямокутника.
        :type width: float
        :param height: Висота прямокутника.
        :type height: float
        """

        self.width = width
        self.height = height

    def area(self):
        """
        Метод area обчислює площу прямокутника.

        :return: Площа прямокутника.
        :rtype: float
        """
        return self.width * self.height

    def perimeter(self):
        """
        Метод perimeter обчислює периметр прямокутника.

        :return: Периметр прямокутника.
        :rtype: float
        """
        return 2 * (self.width + self.height)

    def diagonal(self):
        """
        Метод diagonal обчислює діагональ прямокутника
        :return: Діагональ прямокутника.
        :rtype: float
        """
        return math.sqrt(self.width**2 + self.height**2)


rectangle1 = Rectangle(5, 10)
print(rectangle1.diagonal())


"""Task 2. Створіть клас BankAccount для представлення банківського рахунку. 
Клас повинен мати атрибути account_number, balance та методи для внесення та зняття грошей з рахунку."""


class Account:

    def __init__(self, account_number, balance: float = 0.0):
        self.account_number = account_number
        self.balance = balance

    def __str__(self):
        return f'Account number: {self.account_number}. Balance: {self.balance}'

    def introduction(self, money):
        self.balance += money

    def withdrawal(self, money):
        if money > self.balance:
            print(f"No money in the account {self.account_number} balance {self.balance - money}")
        else:
            self.balance -= money

acc1 = Account('acc_id1', 10.0)

acc1.introduction(50)
acc1.withdrawal(20)
print(acc1)
