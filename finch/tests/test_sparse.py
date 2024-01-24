import pytest
import finch

@pytest.fixture
def x():
    return finch.fsprand(5, 5, 0.5)

@pytest.fixture
def y():
    return finch.fsprand(5, 5, 0.5)

def test_add(x, y):
    print(x.data)
    print(y.data)
    print((x + y).data)
    assert True