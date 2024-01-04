class Command:
    def execute(self):
        ...


class LightOnCommand(Command):
    def __init__(self, light):
        self._light = light

    def execute(self):
        self._light.turn_on()


class LightOffCommand(Command):
    def __init__(self, light):
        self._light = light

    def execute(self):
        self._light.turn_off()


class Light:
    def turn_on(self):
        print('Світло увімкнено.')

    def turn_off(self):
        print('Світло вимкнене.')


class RemoteControl:
    def __init__(self):
        self._command = None

    def set_command(self, command):
        self._command = command

    def press_button(self):
        self._command.execute()


if __name__ == '__main__':
    light = Light()
    ligth_on = LightOnCommand(light)
    ligth_off = LightOffCommand(light)

    remote = RemoteControl()
    remote.set_command(ligth_on)

    remote.press_button()

    remote.set_command(ligth_off)
    remote.press_button()
