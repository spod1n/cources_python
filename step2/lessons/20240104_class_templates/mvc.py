class Model:
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data


class View:
    def display_data(self, data):
        print(f'Display data: {data}')


class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def update_data(self, new_data):
        self.model.data = str(new_data)
        updated_data = self.model.get_data()
        self.view.display_data(updated_data)


if __name__ == '__main__':
    init_data = 'Our Data'

    model = Model(init_data)
    view = View()

    controller = Controller(model, view)
    controller.update_data(init_data)

    new_data = 'New Data'
    controller.update_data(new_data)
