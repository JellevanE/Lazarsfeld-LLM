"""
Configuration file for text evaluation framework.
Define concepts, dimensions, and questions for evaluation.
"""

SAMPLE_EVALUATION_CONFIG = {
    "concepts": [
        {
            "concept_description": "Clarity of Expression",
            "weight": 1.0,
            "dimensions": [
                {
                    "dimension_description": "Sentence Structure",
                    "weight": 0.7,
                    "questions": [
                        {
                            "question": "Are the sentences in this text well-structured with clear subjects and verbs? Answer with True or False.",
                            "positive_contribution": True
                        },
                        {
                            "question": "Does the text contain run-on sentences that are difficult to follow? Answer with True or False.",
                            "positive_contribution": False
                        }
                    ]
                },
                {
                    "dimension_description": "Vocabulary Usage",
                    "weight": 0.5,
                    "questions": [
                        {
                            "question": "Is the vocabulary used in this text appropriate for the intended audience?",
                            "positive_contribution": True
                        },
                        {
                            "question": "Does the text use unnecessarily complex terminology? Answer with True or False.",
                            "positive_contribution": False
                        },
                        {
                            "question": "Do you find the text humorous? And do you think it was written by a machine? Answer with True or False.",
                            "positive_contribution": False
                        }
                    ]
                }
            ]
        },
        {
            "concept_description": "Logical Coherence",
            "weight": 1.2,
            "dimensions": [
                {
                    "dimension_description": "Argument Structure",
                    "weight": 1.0,
                    "questions": [
                        {
                            "question": "Does the text present a clear logical progression of ideas?",
                            "positive_contribution": True
                        },
                        {
                            "question": "Are there logical fallacies or contradictions in the text? Answer with True or False.",
                            "positive_contribution": False
                        }
                    ]
                },
                {
                    "dimension_description": "Evidence Support",
                    "weight": 0.8,
                    "questions": [
                        {
                            "question": "Are claims in the text supported by appropriate evidence? Answer with True or False.",
                            "positive_contribution": True
                        },
                        {
                            "question": "Does the text make unsupported assertions? Answer with True or False.",
                            "positive_contribution": False
                        }
                    ]
                }
            ]
        }
    ],
    "reference_texts": {
        "high_quality": "This is a sample high-quality text that would score well...",
        "medium_quality": "This is a sample medium-quality text with some issues...",
        "low_quality": "This is a poorly written sample with many problems..."
    },
    "models": [
        {
            "model_id": "gpt-4",
            "weight": 0.7,  # weight in final aggregated score
            "parameters": {
                "temperature": 0.1,
                "max_tokens": 50
            }
        },
        {
            "model_id": "gpt-3.5-turbo",
            "weight": 0.3,
            "parameters": {
                "temperature": 0.1,
                "max_tokens": 50
            }
        }
    ],
    "aggregation_method": "weighted_average"  # how to combine scores from different models
}


EPSON_PRINTER_TEXT = """
Hoe kies je een Epson zakelijke printer?
Ben je op zoek naar een Epson zakelijke printer, maar weet je niet welke het beste bij jouw behoeften past? In dit artikel leggen we uit waar je op moet letten. Denk bijvoorbeeld aan automatisch dubbelzijdig scannen en het type inkt. Zo kies jij de beste Epson zakelijke printer voor jouw situatie.
Epson zakelijke printer kiezen Automatisch dubbelzijdig printen Type inkt Maximaal afdrukformaat Afdruksnelheid Formaat printer Inktabonnement
Epson zakelijke printer kiezen
Epson EcoTank op kantoor
Beantwoord deze vragen en vind de beste Epson zakelijke printer voor jou.

Wil je automatisch dubbelzijdig scannen?
Welk type inkt past bij jouw gebruik?
Welk maximaal afdrukformaat heb je nodig?
Hoe snel moet de printer zijn?
Hoe groot mag de printer zijn?
Wil je een inktabonnement?
Bekijk alle Epson printers
Wil je automatisch dubbelzijdig scannen?
Automatische documentinvoer
Via de automatische documentinvoer van de printer scan je ook dubbelzijdig bedrukte documenten in. Met automatisch dubbelzijdig scannen scant de printer automatisch beide kanten van jouw A4. Dit bespaart je veel tijd en handmatig werk. Zonder automatisch dubbelzijdig scannen moet je elk A4'tje zelf omdraaien, wat tijdrovend is en foutgevoelig kan zijn. Kies daarom voor een printer met deze functie als je vaak dubbelzijdige documenten scant.

Welk type inkt past bij jouw gebruik?
inktflesjes
Er zijn 2 verschillende soorten printers, die allebei een ander soort inkt gebruiken. Printers met een inktreservoir vul je bij met inktflesjes. Deze printers hebben een hogere aanschafprijs, maar de flesjes inkt zijn goedkoop. Dit maakt ze geschikt als je vaak en veel kleurendocumenten print. Inkjetprinters die met cartridges printen hebben vaak een goedkopere aanschafprijs, maar de cartridges zijn duurder. Dit maakt ze geschikt als je een goedkope printer zoekt en maar af en toe print.

Welk maximaal afdrukformaat heb je nodig?
A4, A3
Het maximale afdrukformaat is het grootste vel papier waarop de printer print. A4 is met 29,7 bij 21 centimeter het standaard formaat voor documenten. Hierop druk je verslagen, samenvattingen en facturen af. A3 is precies het dubbele formaat van A4, namelijk 29,7 bij 42 centimeter. Je gebruikt A3 vooral voor posters, placemats en blauwdrukken. A3+ is nog groter dan een A3 formaat, met afmetingen van 32,9 bij 48,3 centimeter. A3+ gebruik je vooral voor drukwerk, waarbij je bij A3 formaten ook nog ruimte hebt voor snijmarkeringen.

Wil je snel printen?
epson printer op kantoor
Deze specificatie zegt iets over hoe snel een printer pagina's afdrukt. Printers met een goede snelheidsklasse printen ongeveer 25 pagina's per minuut. Dit is genoeg voor een klein kantoor waar regelmatig geprint wordt. Printers met een gemiddelde snelheidsklasse printen ongeveer 21 pagina's per minuut. Deze printers zijn vooral geschikt voor je thuiswerkplek of kleine kantoren waar niet veel geprint wordt. Een standaard printsnelheid is tot ongeveer 18 pagina's per minuut. Dit is prima voor een printer voor thuis waarmee je af en toe een schoolverslag of e-ticket print.

Hoe groot mag de printer zijn?
formaat printer
Printers hebben verschillende formaten. Een medium printer past op een groot bureau of in een grote kast. Heb je een klein bureau? Dan zet je hem op een aparte tafel. Een grote printer neemt veel ruimte in, waardoor je hem op een aparte tafel of kast zet. Kies een formaat dat past bij de beschikbare ruimte in jouw kantoor of werkplek.

Wil je besparen op je printkosten?
epson readyprint doos op vloermat
Sommige printers zijn geschikt voor een voordelig inktabonnement. Hierbij betaal je een vast bedrag per maand voor een bepaald aantal prints. Als je inkt bijna op is, geeft de printer een seintje aan de fabrikant. Zo krijg je nieuwe cartridges automatisch thuisgestuurd. Met een inktabonnement bespaar je op je printkosten. Losse cartridges zijn duur en de prijzen verschillen veel door de verschillende printmodellen. Met een inktabonnement betaal je een vast bedrag per maand voor het aantal pagina's dat je print."""


BOL_TAFEL_TEXT = """
Productbeschrijving
Eettafel van Mango hout ovaal Naturel

Deze prachtige ovale eettafel is gemaakt van hoogwaardig mangohout en straalt natuurlijke charme uit.
De tafel heeft een dik tafelblad van circa 5 cm, wat zorgt voor een stevige en robuuste uitstraling.
Het natuurlijke mangohout heeft een warme tint die elke eetkamer een gezellige en uitnodigende sfeer geeft.

Kenmerken:

Materiaal: Mangohout
Vorm: Ovaal
Kleur: Naturel
Bladdikte: Circa 5 cm
Afmetingen:

Lengte: 160 cm
Breedte: 90 cm
Hoogte: 77 cm
Deze eettafel is perfect voor zowel dagelijkse maaltijden als gezellige diners met vrienden en familie.
Dankzij de stevige constructie en het hoogwaardige mangohout is deze tafel een duurzame en stijlvolle toevoeging aan je interieur."""