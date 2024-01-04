class Component:
    def operation(self):
        ...


class Leaf(Component):
    def operation(self):
        return 'Листок'


class Composite(Component):
    def __init__(self):
        self.children = []

    def add(self, component):
        self.children.append(component)

    def remove(self, component):
        self.children.remove(component)

    def operation(self):
        results = []

        for child in self.children:
            results.append(child.operation())

        return f"Група: {', '.join(results)}"


if __name__ == '__main__':
    leaf1 = Leaf()
    leaf2 = Leaf()

    composite = Composite()
    composite.add(leaf1)
    composite.add(leaf2)

    root = Composite()
    root.add(composite)
    root.add(Leaf())

    print(root.operation())
