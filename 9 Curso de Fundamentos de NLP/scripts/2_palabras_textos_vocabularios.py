# Clase 1
import nltk
corpus = nltk.corpus.cess_esp.sents()
flatten = [item for sublist in corpus for item in sublist]

# Clase 2
import re

# Meta-caracteres básicos
arr = [w for w in flatten if re.search('es', w)]
print(arr[:5])

# Utilizamos $ para buscar palabras que terminen con un patron
arr = [w for w in flatten if re.search('es$', w)]
print(arr[:5])

# Utilizamos ^ para buscar palabras que inicien con un patron
arr = [w for w in flatten if re.search('^es', w)]
print(arr[:5])

arr = [w for w in flatten if re.search('^..j..t..$', w)]
print(arr)

# Rangos [a-z], [A-Z], [0-9]
arr = [w for w in flatten if re.search('^[ghi][mno][jlk][def]$', w)]
print(arr)

# Clausuras *, * (Kleene closures)
# El comando * es para indicar un 0 o más veces (osea que NO es indispensable que aparezca el patron)
arr = [w for w in flatten if re.search('^(no)*', w)]
print(arr[:10])
# El comando + es para indicar que necesita aparecer al menos 1 vez
arr = [w for w in flatten if re.search('(no)+', w)]
print(arr[:10])


texto = "Mi número de teléfono es 123-456-7890"
patron = r"(\d{3})-(\d{3})-(\d{4})"

coincidencia = re.search(patron, texto)
if coincidencia:
    numero_telefono = coincidencia.group()  # Obtiene la coincidencia completa
    area = coincidencia.group(1)  # Obtiene el grupo de captura 1 (código de área)
    print("Número de teléfono:", numero_telefono)
    print("Código de área:", area)