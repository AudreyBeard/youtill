from youtill import deprecated


@deprecated
def deprec_print():
    print('this is my deprecated function')

def not_deprecated_print():
    print('this is not a deprecated function')


if __name__ == "__main__":
    not_deprecated_print()
    deprec_print()
