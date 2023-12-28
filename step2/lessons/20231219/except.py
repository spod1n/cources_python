class AgeException(Exception):
    def __init__(self, user, age):
        self.user = user
        self.age = age

    def __str__(self):
        return f"User {self.user} can buy alcohol from baba Luba"


class canIByAlcohol:
    def __init__(self, user: str, age: float):
        self.user = user
        self.age = age
        min_age = 18
        if min_age <= self.age:
            pass
        else:
            raise AgeException(self.user, self.age)

    def __str__(self):
        return f"User {self.user} can buy alcohol from baba Luba"


try:
    user_by_alco = canIByAlcohol("Klymentii", 17)
    print(str(user_by_alco))
except AgeException as e:
    print(e)
