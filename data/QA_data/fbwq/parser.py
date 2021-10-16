import json

types = ['train', 'test']

for type in types:
    with open('WebQSP.{}.json'.format(type), 'r') as f, open('pruning_{}.txt'.format(type), 'w') as t:
        data = json.load(f)
        for d in data['Questions']:
            question = d['ProcessedQuestion']
            parses = d['Parses']
            entity = parses[0]['TopicEntityMid']
            chain = []
            for p in parses:
                if p['InferentialChain'] is not None:
                    chain.extend(p['InferentialChain'])
            if len(chain) > 0:
                chain = '|'.join(chain)
                t.write('{} [{}]\t{}\n'.format(question, entity, chain))
