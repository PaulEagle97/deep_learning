import inspect
from nn_fwk import NN

def main():
    # Get the docstring for each method in the class and print it
    for name, func in inspect.getmembers(NN, inspect.isfunction):
        print(f"Function '{name}': {inspect.getdoc(func)}\n")

main()
