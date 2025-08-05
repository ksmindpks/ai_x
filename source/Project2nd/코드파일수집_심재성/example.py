"""이것은 모듈 전체에 대한 설명입니다.
이 모듈은 간단한 계산 기능을 제공합니다.
"""

class Calculator:
    """간단한 계산을 수행하는 클래스입니다."""
    def subtract(self, a, b):
        """두 수의 차를 계산합니다."""
        return a - b

def add(a, b):
    """두 개의 숫자를 더하여 그 결과를 반환합니다.

    Args:
        a (int): 첫 번째 숫자입니다.
        b (int): 두 번째 숫자입니다.

    Returns:
        int: 두 숫자를 더한 결과값입니다.
    """
    return a + b

def multiply(a, b):
    # 이것은 일반 주석이며, Docstring이 아닙니다.
    return a * b

