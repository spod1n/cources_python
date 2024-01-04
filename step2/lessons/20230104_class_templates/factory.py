class Product:
    def create(self):
        raise NotImplementedError


class ConcreteProductA(Product):
    def create(self):
        return 'Product A'


class ConcreteProductB(Product):
    def create(self):
        return 'Product B'


class Creator:
    def factory_method(self):
        raise NotImplementedError

    def get_product(self):
        product = self.factory_method()
        return f'Creator get a {product}'


class ConcreteCreatorA(Creator):
    def factory_method(self):
        return ConcreteProductA()


class ConcreteCreatorB(Creator):
    def factory_method(self):
        return ConcreteProductB()


creator_a = ConcreteCreatorA()
product_a = creator_a.get_product()
print(product_a)

creator_b = ConcreteCreatorB()
product_b = creator_b.get_product()
print(product_b)
