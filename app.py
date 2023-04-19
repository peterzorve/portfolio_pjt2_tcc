
from flask import Flask, render_template, request, flash, redirect, url_for  
from helper_functions_tcc import * 
import numpy as np 

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = "data_uploaded" 
app.secret_key = 'super_secret_key'


classify_dictionary = {}

@app.route('/', methods=['POST', 'GET'])
def pjt3_tcc():
     if request.method == 'POST':
          if request.form['submit_tcc'] == 'text_classifier': 
               max_seq_length, emb_dim = 64, 300
               model = ToxicClassifier()

               train_modeled = torch.load('trained_models/trained_all_model')
               model_state = train_modeled['model_state']
               model = ToxicClassifier(max_seq_len=max_seq_length, emb_dim=emb_dim, hidden=64)
               model.load_state_dict(model_state) 

               user_comment = request.form.get('comment_')

               features = front_padding(encoder(preprocessing(user_comment), fasttext), max_seq_length) 
               
               embeddings = [fasttext.vectors[el] for el in features]

               inputs = torch.stack(  embeddings )

               model.eval()
               with torch.no_grad():
                    prediction = model.forward(inputs.flatten().unsqueeze(1))
                    probability_test = torch.sigmoid(prediction)
                    classes = probability_test > 0.5
               prediction = np.array(classes) 

               # classify_dictionary[user_comment] = ['Toxic', 'Severe_Toxic', 'Obscene', 'Threat', 'Insult', 'Identity_Hate'] 
               classify_dictionary[user_comment] = prediction


     return render_template('pjt3_tcc.html', classify_dictionary_=classify_dictionary) 


if __name__ == '__main__':  
    app.run(debug=True)


