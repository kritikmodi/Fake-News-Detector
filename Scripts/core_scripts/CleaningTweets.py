def cleaningTweet(string):
    # string = clean(string, no_emoji=True)
    string = re.sub(r'http\S+', '', string)
    newlist = []
    for letter in string.split():
        if letter.endswith(":"):
            pass
        else:
            newlist.append(letter)
    string = ' '.join(newlist)
    newlist = []
    for letter in string:
        if letter == "@":
            letter = letter.replace("@", "")
            newlist.append(letter)
        else:
            newlist.append(letter)
    string = ''.join(newlist)
    newstring = string.split()
    res = list(map(lambda st: str.replace(st, "&amp;", "&"), newstring))
    text = ' '.join(res)
    return text
