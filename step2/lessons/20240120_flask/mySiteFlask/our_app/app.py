from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)


@app.route('/objects_many/<string1>')
def objects_many(string1):
    string2 = 'CONSTANTS TEXT'

    df = pd.DataFrame([
        {
            'id': [1, 2, 3],
            'name': ['John', 'Kate', 'Bob']
        }
    ])

    number = df.shape[0]

    return render_template('obj_many.html', text1=string1, text2=string2, table=df, num=number)


if __name__ == '__main__':
    app.run(debug=True)
