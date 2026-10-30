"""Punto 2. Inferencia exacta por enumeración."""
from algorithms.probability import normalize


def enumeration_ask(network, query: str, evidence: dict) -> dict:
    """Calcula P(query | evidence) sobre la conjunta factorizada.

    Entradas: network es la red; query es el nombre de la
    variable consultada; evidence es {nombre de variable: 0 o 1}.
    Salida: {0: probabilidad, 1: probabilidad}.

    Tips:
    - Recorra network.variables en orden topológico.
    - Para una variable fijada multiplique su probabilidad; para una oculta
      sume las dos alternativas 0 y 1. Normalice al terminar.
    - Use copias del evento y no modifique evidence.
    - network.probability(variable, valor, evento) consulta la CPT;
      el evento debe contener los valores de todos los padres.
    - Puede usar una función auxiliar recursiva. Al terminar el recorrido,
      el producto sin factores restantes vale 1.
    - Evidencia imposible: ValueError. Consulta observada: distribución puntual
      solo si la evidencia es posible.
    """
    # Valida nombres de variables y dominio de la evidencia.
    network.validate_query(query, evidence)

    def enumerate_all(variables: list[str], event: dict) -> float:
        # With no variables left, the empty product is 1
        if not variables:
            return 1.0

        first: str = variables[0]
        rest: list[str] = variables[1:]

        # An observed variable multiplies its own probability
        if first in event:
            probability: float = network.probability(first, event[first], event)
            return probability * enumerate_all(rest, event)

        # A hidden variable adds up both of its values, each on a copy of the event
        total = 0.0
        for value in (0, 1):
            extended_event: dict = dict(event)
            extended_event[first] = value
            probability: float = network.probability(first, value, extended_event)
            total += probability * enumerate_all(rest, extended_event)
        return total

    # Weigh each value of the query together with the evidence
    weights: dict = {}
    for value in (0, 1):
        # An observed query only keeps its observed value
        if query in evidence and evidence[query] != value:
            weights[value] = 0.0
            continue

        event: dict = dict(evidence)
        event[query] = value
        weights[value] = enumerate_all(network.variables, event)

    # Evidence with probability zero cannot be conditioned on
    if weights[0] + weights[1] == 0:
        raise ValueError("Evidencia imposible")

    return normalize(weights)
    # FIN DEL BLOQUE DEL ESTUDIANTE
