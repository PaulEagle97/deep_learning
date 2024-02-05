import inspect
from mlp_framework import MLP

def main():
    # Get the docstring for each method in the class and print it
    for name, func in inspect.getmembers(MLP, inspect.isfunction):
        print(f"Function '{name}': {inspect.getdoc(func)}\n")

main()
