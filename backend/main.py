from flask import Flask, Blueprint, jsonify
from  backend.routes import inference 



app = Flask(__name__) # instntiating a flask object


app.register_blueprint(inference)

if __name__ == "__main__":
    app.run(debug = True)
    

