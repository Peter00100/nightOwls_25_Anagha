from django.shortcuts import render
from .models import User

# # Create your views here.


def user(request):
    
    submit=request.POST
    if 'submit' in submit:
        INSTANCE= User(name=submit.get('1'), dob=submit.get('2'), sex=submit.get('3'), email=submit.get('4'), contactno=submit.get('5'), username=submit.get('6'), password=submit.get('7'), address=submit.get('8'))
        INSTANCE.save()
        return render(request, 'userlogin.html')
    return render(request,'users.html')



def userlogin(request):

    submit=request.POST
    
    if 'submit1' in submit:
        uname=str(submit.get('1'))
        passw= str(submit.get('2'))

        if User.objects.filter(username=uname).exists():

            if User.objects.filter(password=passw).exists():

                return render(request,'userlogin.html')
        
        if uname =='admin':
            if passw == 'admin':
                table_data = User.objects.all()  # Fetch all records from your table
                return render(request, 'admin.html', {'table_data': table_data})
    
    if 'submit2' in submit:        
        return render(request, 'users.html')
    
    return render(request,'index.html')

    
def user(request):
    
    submit=request.POST


    if 'submit' in submit:

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import wordcloud
        import warnings
        warnings.filterwarnings('ignore')
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn import metrics
        from sklearn.metrics import accuracy_score
        from pandas.plotting import scatter_matrix
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import metrics

        resumeDataSet = pd.read_csv('/content/UpdatedResumeDataSet.csv' ,encoding='utf-8')
        resumeDataSet['cleaned_resume'] = ''

        import re

        def cleanResume(resumeText):
            resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
            resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
            resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
            resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
            resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
            resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)
            resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
            return resumeText

        resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))
        from pypdf import PdfReader
        def extract_text_with_pyPDF(PDF_File):

            pdf_reader = PdfReader(PDF_File)

            raw_text = ''

            for i, page in enumerate(pdf_reader.pages):

                text = page.extract_text()
                if text:
                    raw_text += text

            return raw_text
        import nltk
        from nltk.corpus import stopwords
        import string
        from wordcloud import WordCloud
        nltk.download('stopwords')
        nltk.download('punkt')

        oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
        totalWords =[]
        Sentences = text_with_pyPDF.split()
        cleanedSentences = ""
        for records in Sentences:
            cleanedText = cleanResume(records)
            cleanedSentences += cleanedText
            requiredWords = nltk.word_tokenize(cleanedText)
            for word in requiredWords:
                if word not in oneSetOfStopWords and word not in string.punctuation:
                    totalWords.append(word)

        wordfreqdist = nltk.FreqDist(totalWords)
        mostcommon = wordfreqdist.most_common(50)
        from sklearn.preprocessing import LabelEncoder

        var_mod = ['Category']
        le = LabelEncoder()
        for i in var_mod:
            resumeDataSet[i] = le.fit_transform(resumeDataSet[i])

        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import hstack
        x=resumeDataSet['Resume']
        y=resumeDataSet['Category']
        requiredText = resumeDataSet['cleaned_resume'].values
        requiredTarget = resumeDataSet['Category'].values

        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            stop_words='english')
        word_vectorizer.fit(requiredText)
        WordFeatures = word_vectorizer.transform(requiredText)


        X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=42, test_size=0.2, shuffle=True, stratify=y)
        
        Categories=np.sort(resumeDataSet['Category'].unique())
        