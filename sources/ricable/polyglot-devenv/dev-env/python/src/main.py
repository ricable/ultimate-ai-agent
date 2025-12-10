"""Simple calculator CLI."""

from calculator import add, divide, multiply, subtract


def main():
    """Main function to run the calculator."""
    print("Simple Calculator")
    print("=================")

    try:
        a = float(input("Enter first number: "))
        b = float(input("Enter second number: "))

        print("\nResults:")
        print(f"{a} + {b} = {add(a, b)}")
        print(f"{a} - {b} = {subtract(a, b)}")
        print(f"{a} * {b} = {multiply(a, b)}")

        if b != 0:
            print(f"{a} / {b} = {divide(a, b)}")
        else:
            print(f"{a} / {b} = Cannot divide by zero")

    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
