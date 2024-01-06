from nltk import ngrams

def tokenizeHumanAnswers(answers):
    tokenizedAnswers = []
    
    for question in answers:
        pom = []
        for answer in question['answers']:
            pom.append(answer.split())
        tokenizedAnswers.append(pom)

    return tokenizedAnswers


def calculateAverageRouge(scorer, hyps, refs):
    avgRouge = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    for ref in refs:
        scores = scorer.score(ref, hyps)
        avgRouge['precision'] += scores['rougeL'].precision
        avgRouge['recall'] += scores['rougeL'].recall
        avgRouge['f1'] += scores['rougeL'].fmeasure

    num_references = len(refs)
    if num_references > 0:
        avgRouge['precision'] /= num_references
        avgRouge['recall'] /= num_references
        avgRouge['f1'] /= num_references

    return avgRouge


def distinct_2_score(response):
    if len(response) < 2:
        return 0.0 

    bigrams = list(ngrams(response, 2))
    distinct_2 = len(set(bigrams)) / len(bigrams)

    return distinct_2 