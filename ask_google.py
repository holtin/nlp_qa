#our project colab
#https://colab.research.google.com/drive/18mY51e5oct8sS6W8m4qaqtOMmd2L00QR

#our project github
#https://github.com/holtin/nlp_qa

import numpy as np # Math
import requests # Getting text from websites
import html2text # Converting wiki pages to plain text
from googlesearch import search # Performing Google searches
import re
from simpletransformers.question_answering import QuestionAnsweringModel
from IPython.display import display
from IPython.html import widgets
from bs4 import BeautifulSoup
from markdown import markdown

#our finetuned model
model = QuestionAnsweringModel('distilbert',"holtin/distilbert-base-uncased-holtin-finetuned-full-squad")

def predict_answer(model, question, contexts, seq_len=600, debug=False):
    split_context = []
    
    if not isinstance(contexts, list):
        contexts = [contexts]
    
    for context in contexts:
        for i in range(0, len(context), seq_len):
            split_context.append(context[i:i+seq_len])
            
    split_context = contexts
    
    f_data = []
    
    for i, c in enumerate(split_context):
        f_data.append(
            {'qas': 
              [{'question': question,
               'id': i,
               'answers': [{'text': ' ', 'answer_start': 0}],
               'is_impossible': True}],
              'context': c
            })
        
    prediction = model.predict(f_data)
    if debug:
        print(prediction[0])
        print(prediction[1])

    maxpos=0
    selectid=0   
    for each_id in prediction[1]:
      if each_id['probability'][0]>maxpos:
        maxpos = each_id['probability'][0]
        selectid=each_id['id']
    if debug:
      print(maxpos)
      print(selectid)
    if(prediction[0][selectid]['answer'][0].strip()!=''):
      return prediction[0][selectid]['answer'][0].strip()
    return 'No answer'

# Source: https://gist.github.com/lorey/eb15a7f3338f959a78cc3661fbc255fe
def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))

    return text

def format_text(text):
    text = markdown_to_text(text)
    text = text.replace('\n', ' ')
    return text

def query_pages(query, n=3):
    return list(search(query, stop=n))

def query_to_text(query, n=3):
    html_conv = html2text.HTML2Text()
    html_conv.ignore_links = True
    html_conv.escape_all = True
    
    #combine all results into one text
    text = ''    
    for link in query_pages(query, n):
        req = requests.get(link)        
        #only handle html, concate all text in to one list
        if "text/html" in req.headers["content-type"]:
          text= text+ (html_conv.handle(req.text))
          text = format_text(text)

    restext=[]
    restext.append(text)    
    return restext

def q_to_a(model, question, n=2, debug=False):
    context = query_to_text(question, n=n)
    pred = predict_answer(model, question, context, debug=debug)
    return pred

def on_button_click(b):
    answer = q_to_a(model, text.value, n=2)
    print('Question:', text.value)
    print('Answer:', answer)
    
text = widgets.Text(description='Question:', width=300)
display(text)

button = widgets.Button(description='Get an Answer')
display(button)

    
button.on_click(on_button_click)