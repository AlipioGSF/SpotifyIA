from flask import Flask, jsonify, request
from IA import recomended


app = Flask(__name__)

@app.route('/recomended/<string:music>', methods=['GET'])
def get_recomended_music(music):
    playlist = recomended(music)
    if playlist is None:
        return jsonify({'error': 'Música não encontrada para gerar uma playlist'}), 404
    return jsonify({'recomended playlist': playlist}), 200

if __name__ == '__main__':
    app.run(debug=True)