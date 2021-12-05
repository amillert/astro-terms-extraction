import nltk
from nltk import RegexpParser
import math

docs = list(map(lambda l: l.strip(), open("out-astro", "r", encoding="utf-8").readlines()))

def term_extraction_rule_based(document, word_num):
    result = []
    document_split = document.split(' ')
    tokens = nltk.word_tokenize(document)
    result_pos_tag = nltk.pos_tag(tokens)

    if word_num == 2:
        pattern = """
            P: {(<CC>|<CD>|<DT>|<EX>|<FW>|<IN>|<JJ>|<JJR>|<JJS>|<MD>|<NN>|<NNS>|<NNP>|<NNPS>|<PDT>|<POS>) (<NN>|<NNP>|<NNS>|<VBJ>|<NNPS>)}
        """
    if word_num == 3:
        pattern = """
            P: {(<CC>|<CD>|<DT>|<EX>|<FW>|<IN>|<JJ>|<JJR>|<JJS>|<MD>|<NN>|<NNS>|<NNP>|<NNPS>|<PDT>|<POS>)(<CC>|<CD>|<DT>|<EX>|<FW>|<IN>|<JJ>|<JJR>|<JJS>|<MD>|<NN>|<NNS>|<NNP>|<NNPS>|<PDT>|<POS>)(<NN>|<NNP>|<NNS>|<VBJ>|<NNPS>)}
        """

    PChunker = RegexpParser(pattern)
    result_extraction = PChunker.parse(result_pos_tag)
    sentece_length = len(document_split)
    print(result_extraction)


    if word_num == 2:
        for np in result_extraction:
            if(str(type(np)) != '<class \'tuple\'>'):
                result_temp = np[0][0] + ' ' + np[1][0]
                if(document.count(result_temp)>0):
                    result.append(result_temp)

    if word_num == 3:
        for np in result_extraction:
            if(str(type(np)) != '<class \'tuple\'>'):
                result_temp = np[0][0] + ' ' + np[1][0] + ' ' + np[2][0]
                if(document.count(result_temp)>0):
                    result.append(result_temp)

    result = set(result)
    result_mass = set(result)

    if word_num == 2:
        for np in result_mass:
            temp = np
            value = math.log((sentece_length * document.count(np)) / (document.count(np.split(' ')[0]) * document.count(np.split(' ')[1])))
            print(np)
            print(value)
            if(value <= 3):
                result.remove(temp)

    if word_num == 3:
        for np in result_mass:
            temp = np
            print(temp)
            value = math.log((document.count(np)/sentece_length) / ((document.count(np.split(' ')[0])/sentece_length)*(document.count(np.split(' ')[1])/sentece_length)*(document.count(np.split(' ')[2])/sentece_length)))
            print(value)
            if (value <= 5):
                result.remove(temp)

    return list(result)

if __name__ == "__main__":

    for index in range(len(docs[:10])):
        result_2 = term_extraction_rule_based(docs[index], 2)
        with open('result/rule_based/' + str(index) + '_2.txt', 'w', encoding='utf-8') as file:
            for term in result_2:
                file.write(term + '\n')

        result_3 = term_extraction_rule_based(docs[index], 3)
        with open('result/rule_based/' + str(index) + '_3.txt', 'w', encoding='utf-8') as file:
            for term in result_3:
                file.write(term + '\n')

    # result_2 = set(result_2)
    # print(result_2)
    # print(len(result_2))
    #
    # result_3 = set(result_3)
    # print(result_3)
    # print(len(result_3))


