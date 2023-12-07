from flask import Flask, jsonify, request
from IA import recomended, mix, get_all_songs

app = Flask(__name__)

@app.route('/recomended/', methods=['POST'])
def get_recomended_music():
    data = request.get_json()
    print(data)
    playlist = recomended(data['track'])
    if playlist is None:
        return jsonify({'error': 'Música não encontrada para gerar uma playlist'}), 404
    return jsonify({'recomended playlist': playlist}), 200

@app.route('/mix/', methods=['POST'])
def get_mix_playlist():
    data = request.get_json()
    print(data)
    playlist = mix(data['track'], data['types'])
    if playlist is None:
        return jsonify({'error': 'Música não encontrada para gerar uma playlist'}), 404
    return jsonify({'recomended playlist': playlist}), 200


@app.route('/songs',methods=['GET'])
def get_songs():
    return jsonify(get_all_songs()), 200


if __name__ == '__main__':
    app.run(debug=True)