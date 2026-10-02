"""Stop-word lists: words too common to help tell documents apart."""

# Kept as blocks of text: a list literal would take one line per word.
_ENGLISH = """
a about above after again against all am an and any are as at be because been before being
below between both but by can did do does doing don down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its
itself just me more most my myself no nor not now of off on once only or other our ours
ourselves out over own s same she should so some such t than that the their theirs them
themselves then there these they this those through to too under until up very was we were
what when where which while who whom why will with you your yours yourself yourselves
"""

_SPANISH = """
a al algo algunas algunos ante antes como con contra cual cuando de del desde donde durante
e el ella ellas ellos en entre era erais eran eras eres es esa esas ese eso esos esta estaba
estaban estado estamos estar estas este esto estos estoy está están fue fueron fui ha había
habían han has hasta hay he la las le les lo los me mi mis mucho muchos muy más mí nada ni
no nos nosotros nuestra nuestras nuestro nuestros o os otra otras otro otros para pero poco
por porque que quien quienes qué se sea sean ser sido sin sobre sois somos son soy su sus
suya suyas suyo suyos sí también tanto te tenemos tener tengo ti tiene tienen todo todos tu
tus tú un una uno unos vosotros y ya yo él
"""

ENGLISH = frozenset(_ENGLISH.split())
SPANISH = frozenset(_SPANISH.split())
