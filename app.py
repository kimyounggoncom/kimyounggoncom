from flask import Flask, render_template, request
from com.kimyounggon.models.titanic.controller import Controller
app = Flask(__name__)

@app.route('/')
def index():
   controller = Controller()
   controller.modeling('train.csv', 'test.csv')
   return render_template("index.html")

@app.route('/intro')
def intro():
   return render_template("titanic.html")




if __name__ == '__main__':
     
   app.run('0.0.0.0',port=5000,debug=True)

   app.config['TEMPLATES_AUTO_RELOAD'] = True