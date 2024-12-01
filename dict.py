dict_en2en_raw = """
undescended, unascended
other ascended, unascended
and embodiment, in embodiment
an embodiment, in embodiment
outer cells, outer selves
separate cells, separate selves
separate cell, separate self
live streams, lifestreams
live stream, lifestream
life streams, lifestreams
life stream, lifestream
burst trauma, birth trauma
subconscious cells, subconscious selves
other cells, other selves
primary cells, primary selves
the cells, the selves
follower bodies, four lower bodies
johan, chohan
a senate master, ascended master
scented master, ascended master
centered master, ascended master
matter light, Ma-ter light
log in, lock in
the first way, the First Ray
the second way, the Second Ray
the third way, the Third Ray
the fourth way, the Fourth Ray
the fifth way, the Fifth Ray
the sixth way, the Sixth Ray
the seventh way, the Seventh Ray
Levenesian, Venetian
Moya, Morya
Siva, Shiva
Moore, More
mother mary, Mother Mary
"""

rdict = { '1000s': 'thousands', 'a collective consciousness': 'the collective consciousness', 'Ascended Master': 'ascended master', 'the ascended master': 'the Ascended Master', 'a golden age': 'the golden age', 'And so ': '', 'And so, ': '', 'and so forth and so on': 'and so forth', "aren't": 'are not', 'behaviour': 'behavior', "can't": 'cannot', 'colour': 'color', 'conscious you': 'Conscious You', "couldn't": 'could not', 'defence': 'defense', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', 'Earth': 'earth', 'flavour': 'flavor', 'freewill': 'free will', 'Golden Age': 'golden age', "haven't": 'have not', "he's": 'he is', "I'm": 'I am', "isn't": 'is not', "it's": 'it is', "It's": 'It is', 'karmic board': 'Karmic Board', 'labour': 'labor', 'offence': 'offense', 'out picture': 'out-picture', 'outpicture': 'out-picture', 'realise': 'realize', 'saviour': 'savior', 'self awareness': 'self-awareness', "shouldn't": 'should not', 'summit lighthouse': 'Summit Lighthouse', "that's": 'that is', "That's": 'That is', "there's": 'there is', "There's": 'There is', "they're": 'they are', "wasn't": 'was not', "we're": 'we are', "We're": 'We are', "weren't": 'were not', "we've": 'we have', "What's": 'What is', "what's": 'what is', "wouldn't": 'would not', "won't": 'will not', "you'll": 'you will', "you're": 'you are', "You're": 'You are', "here's": "here is", "Here's": "Here is", "where's": "where is", "who's": "who is", "Who's": "Who is", "I've": "I have", 'So ': '', 'So, ': '', 'Well ': 'Well, ', "divine plan": "Divine Plan" }


def getDic (dict_raw) :
    new_dict = {}

    lines = dict_raw.split ('\n')
    for line in lines :
        words = line.split (',')
        # print (line, words)
        if len (words) < 2 :
            continue

        dkey = words[0].strip ()
        dval = words[1].strip ()
        if dkey != '' and dval != '' :
            new_dict [dkey] = dval

    return new_dict
###


dict_en2en = getDic (dict_en2en_raw)

# print (dict_en2en)


import re


def word_change (input_str, word_mapping):
    for key, value in word_mapping.items():
        input_str = input_str.replace(key, value)
    return input_str

def word_change_re (input_str, word_mapping):
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(key) for key in word_mapping.keys()) + r')\b')
    return pattern.sub(lambda x: word_mapping[x.group()], input_str)


def mapEn2En (input_str) :
    return word_change (input_str, dict_en2en)
    # return word_change_re (input_str, dict_en2en)


if __name__ == '__main__' :
    text = 'undescended sphere'
    text2 = 'in this dispensation'

    print (mapEn2En (text))
