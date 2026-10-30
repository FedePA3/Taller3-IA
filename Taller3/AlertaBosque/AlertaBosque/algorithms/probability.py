"""Punto 1. Distribuciones discretas y decisiones de respuesta."""
import math
#raise ValueError("")

def normalize(weights: dict) -> dict:
    """Retorna un diccionario nuevo cuyas probabilidades suman uno.

    Entrada: weights = {resultado: peso}.
    Salida: {resultado: probabilidad}, con las mismas claves.

    Tips:
    - Conserve las claves y su orden; no modifique weights.
    - Rechace pesos negativos, no finitos, diccionarios vacíos y suma cero.
    """
    # TODO: Add your code here
    if len(weights) == 0:
        raise ValueError("El diccionario está vacio")
    for y in weights.values():
        if y == None:
            raise ValueError("Los valores tienen que ser diferenetes a None")
        if y < 0:
            raise ValueError("Los valores deberian ser mayores o iguales a 0")
        if not math.isfinite(y):
            raise ValueError("Los valores tienen que ser finitos")
    salida = {}
    total = 0
    for x in weights.values():
        total += x
    if total == 0:
        raise ValueError("La suma de los valores es igual a 0")
    for x in weights:
        salida[x] = weights[x]/total
    return salida
    # FIN DEL BLOQUE DEL ESTUDIANTE
 


def expected_utility(distribution: dict, utilities: dict) -> float:
    """Calcula sum_s P(s) U(s) para una acción.

    Entradas: distribution = {resultado: probabilidad} y
    utilities = {resultado: utilidad} para una sola acción.
    Salida: un número con la utilidad esperada.

    Tips:
    - distribution ya debe estar normalizada; no normalice utilidades.
    - Ambas tablas deben contener los mismos resultados posibles.
    """
    # Validación de las entradas; no modificar.
    if set(distribution) != set(utilities) or not distribution:
        raise ValueError("Probabilidades y utilidades deben tener las mismas claves")
    if any(not math.isfinite(p) or p < 0 for p in distribution.values()):
        raise ValueError("Distribución inválida")
    if not math.isclose(sum(distribution.values()), 1, abs_tol=1e-9):
        raise ValueError("La distribución debe sumar uno")
    if any(not math.isfinite(u) for u in utilities.values()):
        raise ValueError("Utilidades no finitas")
    # TODO: Add your code here
    total = 0
    for x in distribution:
        total += distribution[x] * utilities[x]
    return total
    # FIN DEL BLOQUE DEL ESTUDIANTE


def best_action(distribution: dict, actions: dict) -> tuple[str, dict]:
    """Retorna (nombre de acción, utilidades esperadas por acción).

    Entradas: distribution = {resultado: probabilidad} y
    actions = {acción: {resultado: utilidad}}.
    Salida: (acción elegida, {acción: utilidad esperada}).

    Tips:
    - actions asocia cada acción con su tabla de utilidades.
    - Reutilice expected_utility y conserve la primera acción en los empates.
    """
    # Validación de las acciones; no modificar.
    if not actions:
        raise ValueError("Se requiere al menos una acción")
    # TODO: Add your code here
    Mexp = 0
    dic = {}
    mejor = None
    for x in actions:
        y = expected_utility(distribution,actions[x])
        dic[x] = y
        if mejor is None:
            mejor = x
            Mexp = y
        else:
            if Mexp < y:
                mejor = x
                Mexp = y
    return mejor,dic
    # FIN DEL BLOQUE DEL ESTUDIANTE
