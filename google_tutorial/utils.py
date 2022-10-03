def chars(file):
    unique = []
    f = open(file,'r')
    content = f.read()
    for ch in content:
        if ch not in unique:
            unique.append(ch)
    f.close()
    return unique

PROBLEMATIC_CHARS = '‘’ÁÈÃØ、ÐÕ★“”יגאל כרמון'