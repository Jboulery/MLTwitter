def fr_words(DATA_PATH, candidats) :

    import pandas as pd
    import operator
    import nltk
    from nltk.corpus import stopwords
    
    #nltk stanford french tagger
    from nltk.tag import StanfordPOSTagger
    jar = 'C:/Users/user/Downloads/stanford-postagger-full-2018-02-27/stanford-postagger-full-2018-02-27/stanford-postagger-3.9.1.jar'
    model = 'C:/Users/user/Downloads/stanford-postagger-full-2018-02-27/stanford-postagger-full-2018-02-27/models/french.tagger'
    import os
    java_path = "C:/ProgramData/Oracle/Java/javapath/java.exe"
    os.environ['JAVAHOME'] = java_path
    pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8' )
    
    #tokenizer (enlever les # @ et la ponctuation...)
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    
    #lecture des tweets
    df = pd.read_csv(DATA_PATH)
    df = df[df['text'].notnull()]
    a = len(df)
    fr_words = [[] for i in range (len(candidats))]
    indesirable = ["RT","https","http","c","les", "et", "ça","coach", "ils","thevoice", "quand", "donc","thevoice_tf1" ]
    
    for j in range (len(candidats)):   
        count = dict() 
        candidat = candidats[j]
        for i in range (0,a) :  
            if i in [ 7224, 16457,16458,22348,22349,22350,22351,22352, 22353,22354,22355] : 
                continue 
            else : 
                line = df.at[i,'text']
                tokenized = tokenizer.tokenize(line)
                # ne garder que les mots qui ne sont pas des stop words (de, que, dans...)
                # en minuscule 
                words = [ w.lower() for w in tokenized if (w not in stopwords.words('french') and w not in indesirable)]
                if set(candidat) & set(words):
                    for word in words :
                        
                        if word in count.keys() :
                            count[word] += 1
                        else :
                            count[word] = 1
                else:
                    continue
                
    
        count = sorted(count.items(), key=operator.itemgetter(1), reverse = True)
        
        fr_words1 = count [0:50]
        
        # enlever tous les verbes 
        for element in fr_words1 : 
            if pos_tagger.tag(element[0].split())[0][1] not in ['VINF','V'] :
                fr_words[j].append(element)
            else :
                continue
    return fr_words
    
    
DATA_PATH = 'twitterdump.csv'
candidats = [["ecco","eco"],["solia"],["edouard","eduard"],["aurélien","aurelien"],["rebecca","rébécca","rebeca"],["gulan","gulaan","gulann","gullan","gullaan"],["drea","dréa","dury","duri","dreaduri","dreadury"],["kriil","kril","krill","kriill"],["jat","trio","sarah","sara","ayelya","annabelle"],["jorge","sabelico"],["hobbs","hobs","hobss"],["frederic","frédéric","fréderic","fredéric","longbois","lonbois"],["laura"],["luca"],["mennel","menel","mennell","menell"],["kelly","kelli","Kely","Keli"],["queen","queen","qeen","clairie","claire","clarie"],["abel","marta"],["raffi","rafi","raffy","arto"],["karolyn","carolyn","karolin"],["mélody","melodi","mélodi","melody"],["ritchy","richy","ritchi"],["sherley","sherlei","serley","paredes"],["tiphanie","tipanie","sg","tiphaine","tipaine"],["Lilya","lylia","lilia"],["b","demi-mondaine","demi","mondaine"],["zine","yaala","yala","zyne"],["milena","mylena"],["liv","del","estal"],["simon","simmon","morin"],["gabriel","grabiell"],["guillaume","gillaume","guillame"],["alice","nguyen","gullen","alyce"],["luna","gritt","grit"],["nicolay","nicola","nycolay","sanson"],["casanova","casanoba"],["xam","hurricane"],["ryan","riane","rian","kennedy","kenedy","kennedi"],["thana-marie","thana","marie"], ["anto","antho"], ["juliana", "julianna"], ["betty","betti","patural"], ["ubare"], ["billy","billie","boguard", "bogard"], ["angelo","anjelo"], ["alhan", "alhane"],["maéle","maelle","maele","maélle", "maëlle", "mael"], ["josse", "joss"],["yasmine","jasmine", "amari", "ammari", "yassmine"],["isadora", "isa dora","dora"],["chloé","cloe","cloé","chloe"],["jody","jodi","jodyjody"], ["francé", "francer"],["aliénor","alienor"],["petit green"], ["eric","jettner"],["matthias","matias","mathias", "piaux", "piau"],["yvette", "dantier", "ivette", "yvett", "yvete"], ["morgane", "moreaux", "morau", "maurgane", "moraux"], ["meryem", "meriem", "mariam", "miriam","sassi"],["gabriel", "laurent", "lorent", "gabriel"], ["assia", "assya", "acia"], ["capucine", "capussine"], ["leho", "leo"],["florent", "marchand", "florand", "floren"], ["norig"], ["djeneva","jenega", "geneva"], ["lorah","lorrah","lorra", "cortese"]]

# R = fr_words(DATA_PATH, candidats)

