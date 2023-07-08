from conllu import parse_incr
from pprint import pprint

data_file = open("../data/UD_Spanish-AnCora/es_ancora-ud-dev.conllu", "r", encoding="utf-8")
for tokenlist in parse_incr(data_file):
    print(tokenlist.serialize())
    break

pprint(tokenlist[1])

print(tokenlist[1]['form']+'|'+tokenlist[1]['upos'])
