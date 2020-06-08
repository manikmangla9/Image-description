from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from os import remove

from flask import *  
app = Flask(__name__)

#extract features from each photo in the directory
def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature


def word_for_id(integer,tokenizer):
	for word,index in tokenizer.word_index.items():
		if index==integer:
			return word
	return None

#generate a description for an image
def generate_desc(model,tokenizer,photo,max_length):
	in_text='startseq'
	for i in range(1,max_length):
		sequence=tokenizer.texts_to_sequences([in_text])[0]
		#pad input
		sequence=pad_sequences([sequence],maxlen=max_length)
		#predict next word
		yhat=model.predict([photo,sequence],verbose=0)
		#convert probability to integer
		yhat=argmax(yhat)
		#map integer to word
		word=word_for_id(yhat,tokenizer)
		#stop if we cannot map the word
		if word is None:
			break
		#append as input for generating the next word
		in_text+=' '+word
		#stop if we predict the end of sequence
		if word == 'endseq':
			break
	return in_text

# load the tokenizer



def prediction(filename):
	photo = extract_features(filename)
	tokenizer = load(open('tokenizer.pkl', 'rb'))
	# pre-define the max sequence length (from training)
	max_length = 34
	# load the model
	model = load_model('model_18.h5')
	description = generate_desc(model, tokenizer, photo, max_length)
	#Remove startseq and endseq
	query = description
	stopwords = ['startseq','endseq']
	querywords = query.split()
	resultwords  = [word for word in querywords if word.lower() not in stopwords]
	result = ' '.join(resultwords) 
	return result
 
#print(prediction('sample.jpg'))
@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':
    	f = request.files['file']
    	f.save(f.filename)  
    	result = prediction((f.filename))
    	remove(f.filename)
    	return render_template("success.html", name = result)
        
		
		
		
		
		
		
		
	
            
if __name__ == '__main__':  
    app.run(port=8000,debug = True)  

