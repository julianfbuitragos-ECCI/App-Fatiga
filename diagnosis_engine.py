"""
diagnosis_engine.py
--------------------
Motor de diagnóstico híbrido (enfoque B del diseño): la IA se encarga de
la percepción (clasificar imagen y señal por separado); las REGLAS,
almacenadas en la base de datos, se encargan de traducir esa combinación
en una causa raíz explicable.

Además del cruce imagen + vibración, cuando la fractura es de tipo
"fatiga" se incorpora un CONTEXTO OPERATIVO (temperatura, ambiente, tipo
de carga) para distinguir el subtipo de fatiga (alto ciclo, bajo ciclo,
térmica, por corrosión, por fricción/fretting). Esto es deliberado: esos
subtipos casi no se distinguen por textura o vibración por sí solas, así
que pedírselo al modelo sería forzar una respuesta poco confiable. Es más
honesto (y más acertado) resolverlo con la información de contexto que ya
conoce quien reporta la falla.
"""

from dataclasses import dataclass, field

import database

# Orden de prioridad para resolver el subtipo de fatiga quando hay más de
# un factor de contexto presente a la vez (el más "dominante" primero,
# según cuánto acelera/explica la falla en la literatura de fatiga).
_SUBTYPE_PRIORITY = [
    ("ambiente", "corrosivo", "corrosion"),
    ("carga", "fretting", "fretting"),
    ("temperatura", "alta", "termica"),
    ("carga", "bajo_ciclo", "bajo_ciclo"),
]
_DEFAULT_SUBTYPE = "alto_ciclo"  # el más común si no hay ningún factor especial


def resolve_fatigue_subtype(temperatura: str = "normal", ambiente: str = "normal",
                             carga: str = "alto_ciclo") -> str:
    """Decide el subtipo de fatiga a partir del contexto operativo.

    temperatura: 'normal' | 'alta' | 'baja'
    ambiente:    'normal' | 'corrosivo' | 'abrasivo'
    carga:       'alto_ciclo' | 'bajo_ciclo' | 'fretting'
    """
    context = {"temperatura": temperatura, "ambiente": ambiente, "carga": carga}
    for field_name, value, subtype in _SUBTYPE_PRIORITY:
        if context[field_name] == value:
            return subtype
    return _DEFAULT_SUBTYPE


@dataclass
class DiagnosisResult:
    image_class: str
    image_confidence: float
    vibration_class: str
    vibration_confidence: float
    root_cause: str
    explanation: str
    recommendation: str
    combined_confidence: float
    rule_found: bool
    fatigue_subtype: dict = field(default=None)


def diagnose(image_class: str, image_confidence: float,
             vibration_class: str, vibration_confidence: float,
             temperatura: str = "normal", ambiente: str = "normal",
             carga: str = "alto_ciclo") -> DiagnosisResult:
    """Combina ambas predicciones, busca (o infiere) la causa raíz y,
    si aplica, refina el subtipo de fatiga con el contexto operativo."""

    rule = database.get_rule(image_class, vibration_class)

    if rule:
        root_cause = rule["root_cause"]
        explanation = rule["explanation"]
        recommendation = rule["recommendation"]
        rule_found = True
    else:
        # Sin regla exacta en la base de conocimiento: fallback explicable
        # que combina las descripciones individuales en vez de inventar
        # una causa específica (evita afirmaciones no respaldadas).
        fracture_info = database.get_fracture_type(image_class)
        vibration_info = database.get_vibration_fault(vibration_class)

        root_cause = (
            f"Causa no catalogada específicamente: posible combinación de "
            f"'{fracture_info['name'] if fracture_info else image_class}' con "
            f"'{vibration_info['name'] if vibration_info else vibration_class}'"
        )
        explanation = (
            "No existe una regla específica en la base de conocimiento para "
            "esta combinación exacta. A partir de la información individual: "
            f"{fracture_info['description'] if fracture_info else ''} "
            f"{vibration_info['description'] if vibration_info else ''}"
        ).strip()
        recommendation = (
            "Se recomienda revisión manual por un especialista, dado que la "
            "base de conocimiento no cubre aún esta combinación."
        )
        rule_found = False

    fatigue_subtype_info = None
    if image_class == "fatiga":
        subtype_code = resolve_fatigue_subtype(temperatura, ambiente, carga)
        fatigue_subtype_info = database.get_fatigue_subtype(subtype_code)
        if fatigue_subtype_info:
            root_cause = f"{root_cause} — subtipo: {fatigue_subtype_info['name']}"
            explanation = (
                f"{explanation} Según el contexto operativo indicado, esto "
                f"corresponde específicamente a {fatigue_subtype_info['name'].lower()}: "
                f"{fatigue_subtype_info['description']}"
            )
            recommendation = (
                f"{recommendation} Adicionalmente, dado el subtipo identificado: "
                f"{fatigue_subtype_info['typical_causes']}"
            )

    combined_confidence = round((image_confidence + vibration_confidence) / 2, 3)

    return DiagnosisResult(
        image_class=image_class,
        image_confidence=round(image_confidence, 3),
        vibration_class=vibration_class,
        vibration_confidence=round(vibration_confidence, 3),
        root_cause=root_cause,
        explanation=explanation,
        recommendation=recommendation,
        combined_confidence=combined_confidence,
        rule_found=rule_found,
        fatigue_subtype=fatigue_subtype_info,
    )


if __name__ == "__main__":
    database.init_db()
    result = diagnose("fatiga", 0.93, "pista_interna", 0.88,
                       temperatura="normal", ambiente="corrosivo", carga="alto_ciclo")
    print("Causa raíz:", result.root_cause)
    print("Explicación:", result.explanation)
    print("Recomendación:", result.recommendation)
    print("Confianza combinada:", result.combined_confidence)
