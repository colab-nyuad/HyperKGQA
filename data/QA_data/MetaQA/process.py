types_to_relations = {}
types_to_relations['movie_to_director'] = 'directed_by'
types_to_relations['movie_to_genre'] = 'has_genre'
types_to_relations['genre_to_movie'] = 'has_genre'
types_to_relations['movie_to_language'] = 'in_language'
types_to_relations['movie_to_imdbrating'] = 'has_imdb_rating'
types_to_relations['movie_to_imdbvotes'] = 'has_imdb_votes'
types_to_relations['tag_to_movie'] = 'has_tags'
types_to_relations['movie_to_actor'] = 'starred_actors'
types_to_relations['writer_to_movie'] = 'written_by'
types_to_relations['director_to_movie'] = 'directed_by'
types_to_relations['movie_to_year'] = 'release_year'
types_to_relations['actor_to_movie'] = 'starred_actors'
types_to_relations['movie_to_writer'] = 'written_by'
types_to_relations['movie_to_tags'] = 'has_tags'


types = ['train', 'valid', 'test']

#1 hop
for type in types:
    with open('1hop/{}.txt'.format(type), 'r') as f1, open('1hop/qa_{}_qtype.txt'.format(type), 'r') as f2, open('1hop/pruning_{}.txt'.format(type), 'w') as f:
        for l1, l2 in zip(f1, f2):
            data = l1.strip().split('\t')
            f.write('{}\t{}\n'.format(data[0], types_to_relations[l2.strip()]))

#2 hop
for type in types:
    with open('2hop/{}.txt'.format(type), 'r') as f1, open('2hop/qa_{}_qtype.txt'.format(type), 'r') as f2, open('2hop/pruning_{}.txt'.format(type), 'w') as f:
        for l1, l2 in zip(f1, f2):
            data = l1.strip().split('\t')
            rels = l2.strip().split('_')
            rel1 = '_'.join(rels[:3])
            rel2 = '_'.join(rels[2:])
            f.write('{}\t{}|{}\n'.format(data[0], types_to_relations[rel1], types_to_relations[rel2]))

#3 hop
for type in types:
    with open('3hop/{}.txt'.format(type), 'r') as f1, open('3hop/qa_{}_qtype.txt'.format(type), 'r') as f2, open('3hop/pruning_{}.txt'.format(type), 'w') as f:
        for l1, l2 in zip(f1, f2):
            data = l1.strip().split('\t')
            rels = l2.strip().split('_')
            rel1 = '_'.join(rels[:3])
            rel2 = '_'.join(rels[2:5])
            rel3 = '_'.join(rels[4:])
            f.write('{}\t{}|{}|{}\n'.format(data[0], types_to_relations[rel1], types_to_relations[rel2], types_to_relations[rel3]))


