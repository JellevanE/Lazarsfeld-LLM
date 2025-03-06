from langchain_core.prompts import PromptTemplate


sys_eval_prompt = PromptTemplate.from_template(
    """
Je bent een expert in het beoordelen van teksten. Je bent gespecialiseerd in {concept}. Je beoordeelt teksten op basis van de informatie die je hebt gekregen. Je kunt alleen antwoorden met 'True' of 'False'.
Het is belangrijk dat je neutraal blijft bij de beoordeling.""")

base_eval_prompt = PromptTemplate.from_template(
    """
# Opdracht:
Je krijgt een stuk tekst te zien. Je gaat de tekst evalueren op het gebied van {concept}. Je richt je hierbij op de volgende vraag:
{question}
De vraag draait om de dimensie {dimension} van het concept {concept}. Je mag de vraag alleen beantwoorden met 'True' of 'False'. 
Bekijk de hele tekst goed. Wees hierbij erg kritisch. Geef niet het wenselijke antwoord, maar wees eerlijk. Het is beter om iets te streng te zijn dan te soepel.

# Voorbeeld informatie:
{examples}

# Tekst:
{input_text}


         
# Herhaling vraag:
{question}
         
Evalueer de tekst en beantwoordt de vraag met True of False, wees hierbij kritisch.
""")