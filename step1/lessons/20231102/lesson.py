def test_and_function():
    assert add(5, 5) == 10, 'Error'
    assert add(10, 10) == 20, 'Error'
    assert add('5', '5') == None, 'Concat'

test_and_function()