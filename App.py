# Inclusion of ngrok make it work in google colab like virtual environments
! pip install pyngrok
from pyngrok import ngrok
from flask import Flask,request,render_template
import tensorflow as tf
import pickle
import numpy as np
import nltk
nltk.download('punkt')

port_no=5000

app=Flask(__name__)
ngrok.set_auth_token('2dW7qeQXYf1RbmtHWu1mUriKQnC_6qjCtetiLtbfKXVY6R1ZZ') # Add your Authentication token for ngrok 
public_url = ngrok.connect(port_no).public_url

@app.route('/',methods=['GET'])
def homepage():
  return render_template('index.html')

@app.route('/Bifurcation',methods=['POST'])
def main_function():
  if request.method=='POST':
    some_rct=str(request.form.get('content'))
    from nltk.tokenize import sent_tokenize
    list_of_sent=sent_tokenize(some_rct)

    char_rct_sent=[]
    for i in list_of_sent:
      char_rct=''
      for j in i:
        char_rct=char_rct+' '+i
      char_rct_sent.append(char_rct)

    def total_lines_maker(corpus):
      total_length=len(corpus.split('.'))
      total_length_parameter=[]
      sentence_no=[]
      for i in range(0,len(list_of_sent)):
        total_length_parameter.append(len(list_of_sent))
        sentence_no.append(i)

      return total_length_parameter,sentence_no

    total_lines,sentence_no=total_lines_maker(some_rct)

    loaded_model=tf.keras.models.load_model('Trybrid_model_final')

    results=loaded_model.predict((tf.constant(list_of_sent),tf.constant(char_rct_sent),tf.constant(sentence_no),tf.constant(total_lines)))

    ohe=pickle.load(open('/content/drive/MyDrive/RCT Classification/One Hot Encoder/ohe.pkl','rb'))

    classes=ohe.inverse_transform(results)

    Categories=np.squeeze(classes,axis=1).tolist()

    final_list=[]

    for i,j in zip(Categories,list_of_sent):
      final_list.append({'Sentence':j,'Category':i})


    return render_template('result.html',result=final_list)

  else:
    return render_template('index.html')

print(public_url)

if __name__=='__main__':
  app.run(port=port_no)


