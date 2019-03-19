from youtill import deprecated


@deprecated
def deprec_print():
    print('this is my deprecated function')


if __name__ == "__main__":
    deprec_print()
