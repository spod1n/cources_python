import unittest


class MyTestCase(unittest.TestCase):
    def test1(self):
        self.assertEqual(1 + 1, 2)

    def test2(self):
        self.assertEqual(True, True)

    def test3(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
