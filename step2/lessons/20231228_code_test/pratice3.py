import pytest


def type_letters():
    letters = 'type some let2ers'
    for i in letters:
        if i.isdigit():
            print("hello")
            raise ValueError("Digit found in the string")
        elif i.lower() == i:
            raise ValueError('Found lower symbol')
        elif i.upper() == i:
            raise ValueError('Found upper symbol')
    return 0


def test_type_letters_without_exception():
    assert type_letters() == 0


def test_type_letters_with_exception():
    with pytest.raises(ValueError, match="Digit found in the string"):
        type_letters() == 0


def test_type_letters_lower():
    with pytest.raises(ValueError, match="Found lower symbol"):
        type_letters() == 0


def test_type_letters_upper():
    with pytest.raises(ValueError, match="Found upper symbol"):
        type_letters() == 1


if __name__ == "__main__":
    test_type_letters_without_exception()
    test_type_letters_with_exception()
    test_type_letters_lower()
    test_type_letters_upper()
