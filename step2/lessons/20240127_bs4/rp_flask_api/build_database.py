import datetime as dt

from config import app, db
from models import Person, Note


PEOPLE_NOTES = [
    {
        "lname": "Fairy",
        "fname": "Tooth",
        "notes": [
            ("I brush my teeth after each meal.", "2024-01-25 17:10:24"),
            ("The other day a friend said, I have big teeth.", "2024-01-25 22:17:54"),
            ("Do you pay per gram?", "2022-03-05 22:18:10"),
        ],
    },
    {
        "lname": "Ruprecht",
        "fname": "Knecht",
        "notes": [
            ("I swear, I'll do better this year.", "2024-01-25 09:15:03"),
            ("Really! Only good deeds from now on!", "2024-01-25 13:09:21"),
        ],
    },
    {
        "lname": "Bunny",
        "fname": "Easter",
        "notes": [
            ("Please keep the current inflation rate in mind!", "2024-01-25 22:47:54"),
            ("No need to hide the eggs this time.", "2024-01-25 13:03:17"),
        ],
    },
]

with app.app_context():
    db.drop_all()
    db.create_all()

    for data in PEOPLE_NOTES:
        new_person = Person(lname=data.get('lname'), fname=data.get('fname'))
        for content, timestamp in data.get('notes', []):
            new_person.notes.append(
                Note(
                    content=content,
                    timestamp=dt.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                )
            )
        db.session.add(new_person)
    db.session.commit()
