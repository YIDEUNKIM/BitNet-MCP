from fastmcp import FastMCP
import math
import numpy as np
from scipy import stats
from sympy import symbols, solve, sympify, diff, integrate, oo, Sum
from typing import List, Tuple
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import integrate as sympy_integrate

# Create MCP Server
mcp = FastMCP()

ALLOW_FUNCTION = {
    "math": math,
    "np": np,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "pi": math.pi,
    "e": math.e,
    "abs": abs,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "cot": lambda x: 1 / math.tan(x),
    "csc": lambda x: 1 / math.sin(x),
    "sec": lambda x: 1 / math.cos(x),
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
    "factorial": math.factorial,
    "gamma": math.gamma,
    "erf": math.erf,
    "erfc": math.erfc,
    "lgamma": math.lgamma,
    "degrees": math.degrees,
    "radians": math.radians,
    "isfinite": math.isfinite,
    "isinf": math.isinf,
    "isnan": math.isnan,
    "isqrt": math.isqrt,
    "prod": np.prod,
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "var": np.var,
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
    "cumsum": np.cumsum,
    "cumprod": np.cumprod,
    "clip": np.clip,
    "unique": np.unique,
    "sort": np.sort,
    "argsort": np.argsort,
    "argmax": np.argmax,
}


@mcp.tool()
def calculate(expression: str) -> dict:
    """
    Evaluates a given mathematical expression string. [param: expression, type: math_expression]
    """
    try:
        # Safe evaluation of the expression
        result = eval(
            expression,
            {"__builtins__": {}},
            ALLOW_FUNCTION,
        )
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def solve_equation(equation: str) -> dict:
    """
    Solves an algebraic equation for the variable 'x'. [param: equation, type: equation]
    """
    try:
        x = symbols("x")
        # Split the equation into left and right sides
        parts = equation.split("=")
        if len(parts) != 2:
            return {"error": "Equation must contain an '=' sign"}

        left = sympify(parts[0].strip())
        right = sympify(parts[1].strip())

        # Solve the equation
        solutions = solve(left - right, x)
        return {"solutions": str(solutions)}
    except Exception as e:
        return {"error": str(e)}


# 미분
@mcp.tool()
def differentiate(expression: str, variable: str = "x") -> dict:
    """
    Computes the derivative of an expression. [param: expression, type: math_expression] [param: variable, type: string]
    """
    try:
        var = symbols(variable)
        expr = sympify(expression)
        result = diff(expr, var)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}

#적분
@mcp.tool()
def integrate(expression: str, variable: str = "x") -> dict:
    """
    Computes the indefinite integral of an expression. [param: expression, type: math_expression] [param: variable, type: string]
    For example, integrating '2*x' with respect to 'x' yields 'x**2'.
    """
    try:
        var = symbols(variable)
        expr = sympify(expression)
        result = sympy_integrate(expr, var)  # Use sympy_integrate instead of integrate
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def summation(expression: str, start: int = 0, end: int = 10) -> dict:
    """
    Calculates the summation (sigma) of an expression. [param: expression, type: math_expression] [param: start, type: integer] [param: end, type: integer]
    Requires an expression string, a start value, and an end value.
    """
    try:
        x = sp.Symbol("x")
        expression = sp.sympify(expression)
        summation = sp.Sum(expression, (x, start, end)).doit()
        return {"result": str(summation)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def expand(expression: str) -> dict:
    """
    Expands a mathematical expression. [param: expression, type: math_expression]
    """
    try:
        x = sp.Symbol("x")
        expanded_expression = sp.expand(expression)
        return {"result": str(expanded_expression)}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def factorize(expression: str) -> dict:
    """
    Factorizes a mathematical expression. [param: expression, type: math_expression]
    """
    try:
        x = sp.Symbol("x")
        factored_expression = sp.factor(expression)
        return {"result": str(factored_expression)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting calculator_server")
    # Note: The 'host' and 'port' arguments might cause a TypeError in newer fastmcp versions.
    # If so, use: mcp.run(transport="sse")
    mcp.run(transport="sse", host="localhost", port=8001)